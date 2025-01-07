use std::f32::consts::PI;
use std::marker::PhantomData;

use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use keter::runtime::KernelParameter;

use crate::color::{Color, ColorType, ToFluence};
use crate::fluence::{Fluence, FluenceType, PartialTransmittance, Radiance};
use crate::utils::aabb_intersect;
use crate::{Axis, Grid, Probe};

pub trait Tracer0<F: FluenceType> {
    type Params: KernelParameter;
    fn trace_0(
        &self,
        params: &Self::Params,
        grid: Expr<Grid>,
        cell: Expr<Vec2<i32>>,
        parity: Expr<bool>,
    ) -> [Expr<Fluence<F>>; 2];
}

pub trait Tracer<F: FluenceType> {
    type Params: KernelParameter;
    /// Traces the upper and lower frustrums for a given probe.
    /// TODO: Fill this in
    fn trace(
        &self,
        params: &Self::Params,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> [Expr<Fluence<F>>; 2];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NullTracer;

impl<F: FluenceType> Tracer<F> for NullTracer {
    type Params = ();
    fn trace(
        &self,
        _params: &Self::Params,
        _grid: Expr<Grid>,
        _probe: Expr<Probe>,
    ) -> [Expr<Fluence<F>>; 2] {
        [Fluence::empty().expr(); 2]
    }
}

pub trait WorldTracer<F: FluenceType> {
    type Params: KernelParameter;

    fn trace(
        &self,
        params: &Self::Params,
        start: Expr<Vec2<f32>>,
        direction: Expr<Vec2<f32>>,
        length: Expr<f32>,
    ) -> Expr<Fluence<F>>;
    #[tracked]
    fn trace_to(
        &self,
        params: &Self::Params,
        start: Expr<Vec2<f32>>,
        end: Expr<Vec2<f32>>,
    ) -> Expr<Fluence<F>> {
        let delta = end - start;
        let length = delta.length();
        self.trace(params, start, delta / length, length)
    }
}

// TODO: Deduplicate.
#[derive(Debug, Clone, Copy, PartialEq, Value)]
#[repr(C)]
pub struct WorldSegment {
    pub rotation: Vec2<f32>,
    pub origin: Vec2<f32>,
    pub size: Vec2<f32>,
    pub offset: u32,
}

pub struct SegmentedWorldMapper<F: FluenceType, T> {
    pub tracer: T,
    pub segments: Buffer<WorldSegment>,
    pub _marker: PhantomData<F>,
}
impl<F: FluenceType, T> SegmentedWorldMapper<F, T> {
    // TODO: Redundant.
    #[tracked]
    pub fn to_world_f(
        &self,
        grid: Expr<Grid>,
        cell: Expr<Vec2<f32>>,
        segment: Expr<WorldSegment>,
    ) -> Expr<Vec2<f32>> {
        let cell = cell - Vec2::y() * segment.offset.cast_f32();
        let pos_norm =
            cell / Vec2::expr(grid.size.x, grid.size.y / self.segments.len() as u32).cast_f32();
        let pos_rotated = Vec2::expr(
            pos_norm.x * segment.rotation.x - pos_norm.y * segment.rotation.y,
            pos_norm.x * segment.rotation.y + pos_norm.y * segment.rotation.x,
        );
        pos_rotated * segment.size + segment.origin
    }
    #[tracked]
    pub fn to_world(
        &self,
        grid: Expr<Grid>,
        cell: Expr<Vec2<i32>>,
        segment: Expr<WorldSegment>,
    ) -> Expr<Vec2<f32>> {
        let cell = cell - Vec2::y() * segment.offset.cast_i32();
        let pos_norm = cell.cast_f32()
            / Vec2::expr(grid.size.x, grid.size.y / self.segments.len() as u32).cast_f32();
        let pos_rotated = Vec2::expr(
            pos_norm.x * segment.rotation.x - pos_norm.y * segment.rotation.y,
            pos_norm.x * segment.rotation.y + pos_norm.y * segment.rotation.x,
        );
        pos_rotated * segment.size + segment.origin
    }
    #[tracked]
    pub fn contains(
        &self,
        grid: Expr<Grid>,
        cell: Expr<Vec2<i32>>,
        segment: Expr<WorldSegment>,
    ) -> Expr<bool> {
        (cell.y - segment.offset.cast_i32()).cast_u32() < grid.size.y / self.segments.len() as u32
    }
}
impl<F: FluenceType, T: WorldTracer<F>> SegmentedWorldMapper<F, T> {
    #[tracked]
    pub fn compute_ray(
        &self,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
        tracer_params: &T::Params,
    ) -> Expr<Fluence<F>> {
        let segment = probe.cell.y.cast_u32() / (grid.size.y / self.segments.len() as u32);
        let segment = self.segments.read(segment);
        let start = self.to_world(grid, probe.cell, segment);
        let end_cell = probe.cell + Vec2::expr(1, grid.ray_offset(probe.dir));
        let end = self.to_world(grid, end_cell, segment);
        // TODO: There's a weird bug with the edges that's probably a result of this going over?
        // Also make this extend to infinity?
        self.tracer
            .trace_to(tracer_params, start, end)
            .opaque_if(!self.contains(grid, end_cell, segment))
    }
}

impl<F: FluenceType, T: WorldTracer<F>> Tracer<F> for SegmentedWorldMapper<F, T> {
    type Params = T::Params;
    #[tracked]
    fn trace(
        &self,
        params: &Self::Params,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> [Expr<Fluence<F>>; 2] {
        let next_grid = grid.next();

        let cell = probe.cell;
        let segment = cell.y.cast_u32() / (grid.size.y / self.segments.len() as u32);
        let segment = self.segments.read(segment);
        let dir = probe.dir;

        let start = self.to_world(grid, cell, segment);
        let spacing = (segment.size.x / next_grid.size.x.cast::<f32>())
            / (segment.size.y / (next_grid.size.y / self.segments.len() as u32).cast::<f32>());
        let lower_size = next_grid.angle_size(spacing, dir * 2);
        let upper_size = next_grid.angle_size(spacing, dir * 2 + 1);
        let lower_offset = Vec2::expr(1, grid.lower_offset(dir));
        let upper_offset = Vec2::expr(1, grid.upper_offset(dir));

        let trace = |offset: Expr<Vec2<i32>>| {
            let cell = cell + offset;
            self.tracer
                .trace_to(params, start, self.to_world(grid, cell, segment))
                .opaque_if(!self.contains(grid, cell, segment))
        };

        let factor = (cell.x % 2 == 0).select(2_i32.expr(), 1_i32.expr());
        [
            trace(lower_offset * factor).restrict_angle(lower_size),
            trace(upper_offset * factor).restrict_angle(upper_size),
        ]
    }
}

impl<F: FluenceType, T: WorldTracer<F>> Tracer0<F> for SegmentedWorldMapper<F, T> {
    type Params = (Expr<u32>, T::Params);
    #[tracked]
    fn trace_0(
        &self,
        (segment, params): &Self::Params,
        next_grid: Expr<Grid>,
        cell: Expr<Vec2<i32>>,
        parity: Expr<bool>,
    ) -> [Expr<Fluence<F>>; 2] {
        let cell_f = cell.cast_f32()
            + if parity {
                Vec2::splat_expr(0.0)
            } else {
                Vec2::expr(0.5, 0.5)
            };

        // let segment = cell.y.cast_u32() / (next_grid.size.y / self.segments.len() as u32);
        let segment = self.segments.read(*segment);

        // Counter-correct because of offset
        // actual center-point is at cell - 0.5, so this prevents 2px gaps in shadows.
        // TODO: This is kinda hacky. Also, remove if trying to compute radiance hitting an edge.
        let start = self.to_world_f(next_grid, cell_f - Vec2::expr(0.5, 0.0), segment);

        let lower_size = (PI / 4.0).expr();
        let upper_size = (PI / 4.0).expr();
        let lower_offset = Vec2::expr(0.5, -0.5);
        let upper_offset = Vec2::expr(0.5, 0.5);

        let factor = if parity { 2.0.expr() } else { 1.0.expr() };

        let lower = (cell_f + lower_offset * factor).cast_i32();
        let upper = (cell_f + upper_offset * factor).cast_i32();

        [
            self.tracer
                .trace_to(params, start, self.to_world(next_grid, lower, segment))
                .opaque_if(!self.contains(next_grid, lower, segment))
                .restrict_angle(lower_size),
            self.tracer
                .trace_to(params, start, self.to_world(next_grid, upper, segment))
                .opaque_if(!self.contains(next_grid, upper, segment))
                .restrict_angle(upper_size),
        ]
    }
}

pub struct WorldMapper<F: FluenceType, T> {
    pub tracer: T,
    pub rotation: Vec2<f32>,
    pub world_origin: Vec2<f32>,
    pub world_size: Vec2<f32>,
    pub _marker: PhantomData<F>,
}
impl<F: FluenceType, T> WorldMapper<F, T> {
    #[tracked]
    pub fn to_world(&self, grid: Expr<Grid>, cell: Expr<Vec2<i32>>) -> Expr<Vec2<f32>> {
        let pos_norm = cell.cast_f32() / grid.size.cast_f32() - 0.5;
        let pos_rotated = Vec2::expr(
            pos_norm.x * self.rotation.x - pos_norm.y * self.rotation.y,
            pos_norm.x * self.rotation.y + pos_norm.y * self.rotation.x,
        );
        pos_rotated * self.world_size + self.world_origin
    }
}

impl<F: FluenceType, T: WorldTracer<F>> Tracer<F> for WorldMapper<F, T> {
    type Params = T::Params;
    #[tracked]
    fn trace(
        &self,
        params: &Self::Params,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> [Expr<Fluence<F>>; 2] {
        let next_grid = grid.next();

        let cell = probe.cell;
        let dir = probe.dir;

        let start = self.to_world(grid, cell);
        let spacing = (self.world_size.x / next_grid.size.x.cast::<f32>())
            / (self.world_size.y / next_grid.size.y.cast::<f32>());
        let lower_size = next_grid.angle_size(spacing, dir * 2);
        let upper_size = next_grid.angle_size(spacing, dir * 2 + 1);
        let lower_offset = Vec2::expr(1, grid.lower_offset(dir));
        let upper_offset = Vec2::expr(1, grid.upper_offset(dir));

        if cell.x % 2 == 0 {
            [
                self.tracer
                    .trace_to(params, start, self.to_world(grid, cell + lower_offset * 2))
                    .restrict_angle(lower_size),
                self.tracer
                    .trace_to(params, start, self.to_world(grid, cell + upper_offset * 2))
                    .restrict_angle(upper_size),
            ]
        } else {
            [
                self.tracer
                    .trace_to(params, start, self.to_world(grid, cell + lower_offset))
                    .restrict_angle(lower_size),
                self.tracer
                    .trace_to(params, start, self.to_world(grid, cell + upper_offset))
                    .restrict_angle(upper_size),
            ]
        }
    }
}

mod block;
use block::Block;

type BlockType = u64;

pub struct VoxelTracer<C: ColorType>
where
    C::Emission: IoTexel,
    C::Opacity: IoTexel,
{
    pub emission: Tex2d<C::Emission>,
    pub opacity: Tex2d<C::Opacity>,
    pub diff: Tex2d<<BlockType as Block>::Storage>,
    pub diff_blocks: Tex2d<bool>,
    pub size: Vec2<u32>,
}
impl<C: ColorType> VoxelTracer<C>
where
    C::Emission: IoTexel,
    C::Opacity: IoTexel,
{
    pub fn new(size: Vec2<u32>) -> Self {
        Self {
            // TODO: Un-hardcode.
            emission: DEVICE.create_tex2d::<C::Emission>(PixelStorage::Half4, size.x, size.y, 1),
            opacity: DEVICE.create_tex2d::<C::Opacity>(PixelStorage::Half4, size.x, size.y, 1),
            diff: DEVICE.create_tex2d::<<BlockType as Block>::Storage>(
                BlockType::STORAGE_FORMAT,
                size.x / BlockType::SIZE,
                size.y / BlockType::SIZE,
                1,
            ),
            diff_blocks: DEVICE.create_tex2d::<bool>(
                PixelStorage::Byte1,
                size.x / BlockType::SIZE,
                size.y / BlockType::SIZE,
                1,
            ),
            size,
        }
    }
}
const TRANSMITTANCE_CUTOFF: f16 = f16::from_f32_const(0.001);

impl VoxelTracer<crate::color::RgbF16> {
    #[tracked]
    pub fn compute_diff(&self) {
        let block = BlockType::empty().var();
        for dx in 0..BlockType::SIZE {
            for dy in 0..BlockType::SIZE {
                let pos = dispatch_id().xy() * BlockType::SIZE + Vec2::expr(dx, dy);
                let diff = false.var();
                let this_emission = self.emission.read(pos);
                let this_opacity = self.opacity.read(pos);
                for i in 0_u32..4_u32 {
                    let offset = [
                        Vec2::new(1, 0),
                        Vec2::new(-1, 0),
                        Vec2::new(0, 1),
                        Vec2::new(0, -1),
                    ]
                    .expr()[i];
                    let neighbor = pos.cast_i32() + offset;
                    if (neighbor >= 0).all() && (neighbor < self.size.expr().cast_i32()).all() {
                        let neighbor_emission = self.emission.read(neighbor.cast_u32());
                        let neighbor_opacity = self.opacity.read(neighbor.cast_u32());
                        if (neighbor_emission != this_emission).any()
                            || (neighbor_opacity != this_opacity).any()
                        {
                            *diff = true;
                            break;
                        }
                    }
                }
                if diff {
                    BlockType::set(block, Vec2::expr(dx, dy));
                }
            }
        }
        self.diff_blocks
            .write(dispatch_id().xy(), !BlockType::is_empty(**block));
        BlockType::write(&self.diff.view(0), dispatch_id().xy(), **block);
    }
    #[tracked]
    pub fn trace_opt(
        &self,
        start: Expr<Vec2<f32>>,
        ray_dir: Expr<Vec2<f32>>,
        length: Expr<f32>,
    ) -> Expr<Fluence<crate::fluence::RgbF16>> {
        let start = start + Vec2::new(0.01, 0.01);
        let inv_dir = (ray_dir + f32::EPSILON).recip();

        let interval = aabb_intersect(
            start,
            inv_dir,
            Vec2::splat(0.1).expr(),
            self.size.expr().cast_f32() - Vec2::splat(0.1).expr(),
        );
        let start_t = keter::max(interval.x, 0.0);
        let ray_start = start + start_t * ray_dir;
        let end_t = keter::min(interval.y, length) - start_t;
        if end_t <= 0.01 {
            Fluence::empty().expr()
        } else {
            let pos = ray_start.floor().cast_u32().var();

            let delta_dist = inv_dir.abs();
            let block_delta_dist = delta_dist * BlockType::SIZE as f32;

            let ray_step = ray_dir.signum().cast_i32().cast_u32();
            let side_dist =
                (ray_dir.signum() * (pos.cast_f32() - ray_start) + ray_dir.signum() * 0.5 + 0.5)
                    * delta_dist;
            let side_dist = side_dist.var();

            let block_offset = (ray_dir > 0.0).select(
                Vec2::splat_expr(0_u32),
                Vec2::splat_expr(BlockType::SIZE - 1),
            );

            let last_t = 0.0_f32.var();
            let fluence = Fluence::<crate::fluence::RgbF16>::empty().var();

            let finished = false.var();

            loop {
                loop {
                    let next_t = side_dist.reduce_min();

                    let block = BlockType::read(&self.diff.view(0), pos / BlockType::SIZE);

                    if BlockType::is_empty(block) {
                        break;
                    }

                    if BlockType::get(block, pos % BlockType::SIZE) || next_t >= end_t {
                        let segment_size = keter::min(next_t, end_t) - last_t;
                        let color = Color::<crate::color::RgbF16>::expr(
                            self.emission.read(pos),
                            self.opacity.read(pos),
                        );
                        *fluence = fluence.over(color.to_fluence(segment_size));

                        *last_t = next_t;

                        if (fluence.transmittance < TRANSMITTANCE_CUTOFF).all() {
                            *fluence.transmittance = Vec3::splat(f16::ZERO);
                            *finished = true;
                            break;
                        }

                        if next_t >= end_t {
                            *finished = true;
                            break;
                        }
                    }

                    let mask = side_dist <= side_dist.yx();

                    *side_dist += mask.select(delta_dist, Vec2::splat_expr(0.0));
                    *pos += mask.select(ray_step, Vec2::splat_expr(0));
                }

                if finished {
                    break;
                }

                let block_pos = (pos / BlockType::SIZE).var();
                let block_side_dist = (ray_dir.signum()
                    * (block_pos.cast_f32() - ray_start / BlockType::SIZE as f32)
                    + ray_dir.signum() * 0.5
                    + 0.5)
                    * block_delta_dist;
                let block_side_dist = block_side_dist.var();

                let next_t = block_side_dist.reduce_min().var();

                loop {
                    if next_t >= end_t {
                        let segment_size = end_t - last_t;
                        let color = Color::<crate::color::RgbF16>::expr(
                            self.emission.read(pos),
                            self.opacity.read(pos),
                        );
                        *fluence = fluence.over(color.to_fluence(segment_size));

                        *finished = true;
                        break;
                    }

                    let mask = block_side_dist <= block_side_dist.yx();

                    *block_side_dist += mask.select(block_delta_dist, Vec2::splat_expr(0.0));
                    *block_pos += mask.select(ray_step, Vec2::splat_expr(0));

                    let last_t = **next_t;
                    *next_t = block_side_dist.reduce_min();

                    if self.diff_blocks.read(block_pos) {
                        *pos = mask.select(
                            block_pos * BlockType::SIZE + block_offset,
                            (last_t * ray_dir + ray_start).floor().cast_u32(),
                        );
                        // let a = (pos / B::SIZE == block_pos).all();
                        // lc_assert!(a);
                        // This bugfix is necessary due to floating point issues.
                        if (pos / BlockType::SIZE != block_pos).any() {
                            // *fluence = Fluence::black();
                            *finished = true;
                        }
                        *side_dist = (ray_dir.signum() * (pos.cast_f32() - ray_start)
                            + ray_dir.signum() * 0.5
                            + 0.5)
                            * delta_dist;

                        break;
                    }
                }

                if finished {
                    break;
                }
            }
            **fluence
        }
    }
}
impl<C: ColorType, F: FluenceType> WorldTracer<F> for VoxelTracer<C>
where
    C: ToFluence<F>,
    C::Emission: IoTexel,
    C::Opacity: IoTexel,
{
    type Params = ();
    #[tracked]
    fn trace(
        &self,
        _params: &(),
        start: Expr<Vec2<f32>>,
        ray_dir: Expr<Vec2<f32>>,
        length: Expr<f32>,
    ) -> Expr<Fluence<F>> {
        let start = start + Vec2::new(0.01, 0.01);
        let inv_dir = (ray_dir + f32::EPSILON).recip();
        let interval = aabb_intersect(
            start,
            inv_dir,
            Vec2::splat(0.1).expr(),
            self.size.expr().cast_f32() - Vec2::splat(0.1).expr(),
        );
        let start_t = keter::max(interval.x, 0.0);
        let ray_start = start + start_t * ray_dir;
        let end_t = keter::min(interval.y, length) - start_t;
        if end_t <= 0.01 {
            Fluence::<F>::empty().expr()
        } else {
            let pos = ray_start.floor().cast_u32().var();

            let delta_dist = inv_dir.abs();

            let ray_step = ray_dir.signum().cast_i32().cast_u32();
            let side_dist =
                (ray_dir.signum() * (pos.cast_f32() - ray_start) + ray_dir.signum() * 0.5 + 0.5)
                    * delta_dist;
            let side_dist = side_dist.var();

            let last_t = 0.0_f32.var();

            let fluence = Fluence::<F>::empty().var();

            loop {
                let next_t = side_dist.reduce_min();
                let color = Color::<C>::expr(self.emission.read(pos), self.opacity.read(pos));
                let segment_size = keter::min(next_t, end_t) - last_t;
                *fluence = fluence.over(color.to_fluence(segment_size));
                *last_t = next_t;
                if next_t >= end_t {
                    break;
                }
                let mask = side_dist <= side_dist.yx();

                *side_dist += mask.select(delta_dist, Vec2::splat_expr(0.0));
                *pos += mask.select(ray_step, Vec2::splat_expr(0));
            }

            **fluence
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Value)]
pub struct Circle<R: Radiance> {
    pub center: Vec2<f32>,
    pub radius: f32,
    pub color: R,
}

#[tracked]
fn intersect_circle(
    ray_start: Expr<Vec2<f32>>,
    ray_dir: Expr<Vec2<f32>>,
    radius: Expr<f32>,
) -> (Expr<f32>, Expr<f32>, Expr<f32>, Expr<bool>) {
    let dist_to_parallel = -ray_start.dot(ray_dir);
    let min_point = ray_start + dist_to_parallel * ray_dir;
    let dist_to_center = min_point.length();
    let penetration = radius - dist_to_center;
    if penetration < 0.0.expr() {
        (0.0.expr(), 0.0.expr(), penetration, false.expr())
    } else {
        let dist_to_intersection = (radius.sqr() - dist_to_center.sqr()).sqrt();
        let min_t = dist_to_parallel - dist_to_intersection;
        let max_t = dist_to_parallel + dist_to_intersection;
        (min_t, max_t, penetration, true.expr())
    }
}

pub struct AnalyticTracer<F: FluenceType> {
    pub buffer: Buffer<Circle<F::Radiance>>,
}

impl<F: FluenceType> WorldTracer<F> for AnalyticTracer<F> {
    type Params = ();
    #[tracked]
    fn trace(
        &self,
        _params: &Self::Params,
        start: Expr<Vec2<f32>>,
        dir: Expr<Vec2<f32>>,
        end_t: Expr<f32>,
    ) -> Expr<Fluence<F>> {
        let best_t = end_t.var();
        let best_color = F::Radiance::black().var();
        for i in 0_u32.expr()..self.buffer.len_expr_u32() {
            let circle = self.buffer.read(i);
            let center = circle.center;
            let radius = circle.radius;
            let color = circle.color;
            let (min_t, max_t, penetration, hit) = intersect_circle(start - center, dir, radius);
            if hit && max_t > 0.0 && min_t < best_t {
                *best_t = min_t;
                // TODO: Make this adjustable / always stay at the x-axis size or something.
                // Also make it part of the opacity instead.
                *best_color = F::Radiance::scale(color, keter::min(penetration / 2.0, 1.0));
            }
        }
        if best_t < end_t {
            Fluence::solid_expr(**best_color)
        } else {
            Fluence::empty().expr()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageTracer {
    pub axes: [Axis; 3],
}
impl StorageTracer {
    #[tracked]
    pub fn load<F: FluenceType>(
        &self,
        buffer: &BufferVar<Fluence<F>>,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> Expr<Fluence<F>> {
        let cell = probe.cell.cast_u32();
        buffer.read(Axis::join(
            self.axes,
            (cell, probe.dir),
            (grid.size, grid.directions + 1),
        ))
    }
    #[tracked]
    pub fn load_opt<F: FluenceType>(
        &self,

        buffer: &BufferVar<Fluence<F>>,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> Expr<Fluence<F>> {
        let cell = probe.cell.cast_u32();
        if (cell >= grid.size).any() {
            Fluence::empty().expr()
        } else {
            buffer.read(Axis::join(
                self.axes,
                (cell, probe.dir),
                (grid.size, grid.directions + 1),
            ))
        }
    }
    #[tracked]
    pub fn store<F: FluenceType>(
        &self,
        buffer: &BufferVar<Fluence<F>>,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
        fluence: Expr<Fluence<F>>,
    ) {
        let cell = probe.cell.cast_u32();
        buffer.write(
            Axis::join(
                self.axes,
                (cell, probe.dir),
                (grid.size, grid.directions + 1),
            ),
            fluence,
        );
    }
}

impl<F: FluenceType> Tracer<F> for StorageTracer {
    type Params = (BufferVar<Fluence<F>>, BufferVar<Fluence<F>>, Expr<f32>);
    #[tracked]
    fn trace(
        &self,
        (buffer, next_buffer, spacing): &Self::Params,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> [Expr<Fluence<F>>; 2] {
        // TODO: Can compress if doing bindless or something.
        let next_grid = grid.next();
        let lower_size = next_grid.angle_size(*spacing, probe.dir * 2);
        let upper_size = next_grid.angle_size(*spacing, probe.dir * 2 + 1);
        if probe.cell.x % 2 == 0 {
            [
                self.load(next_buffer, next_grid, probe.next().with_dir(probe.dir * 2))
                    .restrict_angle(lower_size),
                self.load(
                    next_buffer,
                    next_grid,
                    probe.next().with_dir((probe.dir + 1) * 2),
                )
                .restrict_angle(upper_size),
            ]
        } else {
            let a = buffer.len_expr_u32() >= (grid.size.x * grid.size.y * (grid.directions + 1));
            lc_assert!(a);
            [
                self.load(buffer, grid, probe).restrict_angle(lower_size),
                self.load(buffer, grid, probe.with_dir(probe.dir + 1))
                    .restrict_angle(upper_size),
            ]
        }
    }
}

// TODO: try to prevent integer division / multiplication
#[tracked]
pub fn merge_up<F: FluenceType>(
    storage: &StorageTracer,
    last_buffer: &BufferVar<Fluence<F>>,
    grid: Expr<Grid>,
    probe: Expr<Probe>,
) -> Expr<Fluence<F>>
where
    F::Transmittance: PartialTransmittance,
{
    let cell = probe.cell;
    let dir = probe.dir;

    let last_grid = grid.last();
    let last_cell = Vec2::expr(cell.x * 2, cell.y);
    let offset = grid.ray_offset(dir);
    if dir % 2 == 0 {
        let last_dir = dir / 2;
        let midpoint = Probe::expr(
            // TODO: integer division ow
            last_cell + Vec2::expr(1, offset / 2),
            last_dir,
        );

        F::over(
            storage.load(last_buffer, last_grid, Probe::expr(last_cell, last_dir)),
            storage.load_opt(last_buffer, last_grid, midpoint),
        )
    } else {
        let last_dir_lower = dir / 2;
        let last_dir_upper = dir / 2 + 1;
        // TODO: Can simplify by expanding offset.
        let midpoint_lower = Probe::expr(
            last_cell + Vec2::expr(1, (offset.cast_f32() / 2.0).floor().cast_i32()),
            last_dir_upper,
        );
        let midpoint_upper = Probe::expr(
            last_cell + Vec2::expr(1, (offset.cast_f32() / 2.0).ceil().cast_i32()),
            last_dir_lower,
        );

        Fluence::blend(
            F::over(
                storage.load(
                    last_buffer,
                    last_grid,
                    Probe::expr(last_cell, last_dir_lower),
                ),
                storage.load_opt(last_buffer, last_grid, midpoint_lower),
            ),
            F::over(
                storage.load(
                    last_buffer,
                    last_grid,
                    Probe::expr(last_cell, last_dir_upper),
                ),
                storage.load_opt(last_buffer, last_grid, midpoint_upper),
            ),
        )
    }
}

use std::marker::PhantomData;

use keter::lang::types::vector::Vec2;
use keter::prelude::*;
use keter::runtime::{AsKernelArg, KernelParameter};

use crate::color::{Fluence, MergeFluence, PartialTransmittance, Radiance};
use crate::{Grid, Probe};

pub trait Tracer<F: MergeFluence> {
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

pub struct NullTracer;

impl<F: MergeFluence> Tracer<F> for NullTracer {
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

pub trait WorldTracer<F: MergeFluence> {
    type Params: KernelParameter;

    fn trace(
        &self,
        params: &Self::Params,
        start: Expr<Vec2<f32>>,
        end: Expr<Vec2<f32>>,
    ) -> Expr<Fluence<F>>;
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

pub struct SegmentedWorldMapper<F: MergeFluence, T> {
    pub tracer: T,
    pub segments: Buffer<WorldSegment>,
    pub _marker: PhantomData<F>,
}
impl<F: MergeFluence, T> SegmentedWorldMapper<F, T> {
    #[tracked]
    pub fn to_world(
        &self,
        grid: Expr<Grid>,
        cell: Expr<Vec2<i32>>,
        segment: Expr<WorldSegment>,
    ) -> Expr<Vec2<f32>> {
        let cell = cell - Vec2::y() * segment.offset.cast_i32();
        let pos_norm = cell.cast_f32()
            / Vec2::expr(grid.size.x, grid.size.y / self.segments.len() as u32).cast_f32()
            - 0.5;
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
impl<F: MergeFluence, T: WorldTracer<F>> SegmentedWorldMapper<F, T> {
    pub fn cache_level(
        &self,
        grid: Grid,
        buffer: &Buffer<Fluence<F>>,
        args: &impl AsKernelArg<Output = <T::Params as KernelParameter>::Arg>,
    ) {
        assert!(buffer.len() as u32 >= grid.size.x * grid.size.y * (grid.directions + 1));
        let kernel = DEVICE.create_kernel::<fn(<T::Params as KernelParameter>::Arg)>(&track!(
            |tracer_params| {
                let grid = grid.expr();
                let cell = dispatch_id().xy();
                let segment = cell.y / (grid.size.y / self.segments.len() as u32);
                let segment = self.segments.read(segment);
                let cell = cell.cast_i32();
                let dir = dispatch_id().z;
                let start = self.to_world(grid, cell, segment);
                let end_cell = cell + Vec2::expr(1, grid.lower_offset(dir));
                let end = self.to_world(grid, end_cell, segment);
                let fluence = self
                    .tracer
                    .trace(&tracer_params, start, end)
                    .opaque_if(!self.contains(grid, end_cell, segment));
                StorageTracer::store(&buffer.var(), grid, Probe::expr(cell, dir), fluence);
            }
        ));
        kernel.dispatch([grid.size.x, grid.size.y, grid.directions + 1], args);
    }
}

impl<F: MergeFluence, T: WorldTracer<F>> Tracer<F> for SegmentedWorldMapper<F, T> {
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
                .trace(params, start, self.to_world(grid, cell, segment))
                .opaque_if(!self.contains(grid, cell, segment))
        };

        // TODO: Can join.
        if cell.x % 2 == 0 {
            [
                trace(lower_offset * 2).restrict_angle(lower_size),
                trace(upper_offset * 2).restrict_angle(upper_size),
            ]
        } else {
            [
                trace(lower_offset).restrict_angle(lower_size),
                trace(upper_offset).restrict_angle(upper_size),
            ]
        }
    }
}

pub struct SegmentedFrustrumWorldMapper<F: MergeFluence, T> {
    pub tracer: T,
    pub segments: Buffer<WorldSegment>,
    pub frustrums: u32,
    pub _marker: PhantomData<F>,
}
impl<F: MergeFluence, T> SegmentedFrustrumWorldMapper<F, T> {
    #[tracked]
    pub fn to_world(
        &self,
        grid: Expr<Grid>,
        cell: Expr<Vec2<f32>>,
        segment: Expr<WorldSegment>,
    ) -> Expr<Vec2<f32>> {
        let cell = cell - Vec2::y() * segment.offset.cast_f32();
        let pos_norm = cell
            / Vec2::expr(grid.size.x, grid.size.y / self.segments.len() as u32).cast_f32()
            - 0.5;
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
        cell: Expr<Vec2<f32>>,
        segment: Expr<WorldSegment>,
    ) -> Expr<bool> {
        let c = cell.y - segment.offset.cast_f32();
        c >= 0.0 && c < (grid.size.y / self.segments.len() as u32).cast_f32()
    }
}

impl<F: MergeFluence, T: WorldTracer<F>> Tracer<F> for SegmentedFrustrumWorldMapper<F, T>
where
    F::Transmittance: PartialTransmittance,
{
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

        let start = cell.cast_f32();
        let upper = cell.cast_f32() + Vec2::y() * 0.5;
        let lower = cell.cast_f32() - Vec2::y() * 0.5;
        let end1 = cell.cast_f32() + Vec2::expr(1.0, grid.offset(dir.cast_f32()));
        let end2 = cell.cast_f32() + 2.0 * Vec2::expr(1.0, grid.offset(dir.cast_f32()));
        let spacing = (segment.size.x / next_grid.size.x.cast::<f32>())
            / (segment.size.y / (next_grid.size.y / self.segments.len() as u32).cast::<f32>());
        let lower_size = next_grid.angle_size(spacing, dir * 2);
        let upper_size = next_grid.angle_size(spacing, dir * 2 + 1);
        let lower_offset = Vec2::expr(1, grid.lower_offset(dir));
        let upper_offset = Vec2::expr(1, grid.upper_offset(dir));

        let trace =
            |offset: Expr<Vec2<i32>>, other_start: Expr<Vec2<f32>>, end: Expr<Vec2<f32>>| {
                let other_end = cell + offset;
                Fluence::blend_n(
                    &(0..self.frustrums)
                        .map(|i| {
                            let a = i as f32 / (self.frustrums - 1) as f32;
                            let b = 1.0 - a;
                            let s = other_start * a + start * b;
                            let e = end * a + other_end.cast_f32() * b;
                            self.tracer
                                .trace(
                                    params,
                                    self.to_world(grid, s, segment),
                                    self.to_world(grid, e, segment),
                                )
                                .opaque_if(!self.contains(grid, e, segment))
                        })
                        .collect::<Vec<_>>(),
                )
            };

        if cell.x % 2 == 0 {
            [
                trace(lower_offset * 2, lower, end2).restrict_angle(lower_size),
                trace(upper_offset * 2, upper, end2).restrict_angle(upper_size),
            ]
        } else {
            [
                trace(lower_offset, lower, end1).restrict_angle(lower_size),
                trace(upper_offset, upper, end1).restrict_angle(upper_size),
            ]
        }
    }
}

pub struct WorldMapper<F: MergeFluence, T> {
    pub tracer: T,
    pub rotation: Vec2<f32>,
    pub world_origin: Vec2<f32>,
    pub world_size: Vec2<f32>,
    pub _marker: PhantomData<F>,
}
impl<F: MergeFluence, T> WorldMapper<F, T> {
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

impl<F: MergeFluence, T: WorldTracer<F>> Tracer<F> for WorldMapper<F, T> {
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
                    .trace(params, start, self.to_world(grid, cell + lower_offset * 2))
                    .restrict_angle(lower_size),
                self.tracer
                    .trace(params, start, self.to_world(grid, cell + upper_offset * 2))
                    .restrict_angle(upper_size),
            ]
        } else {
            [
                self.tracer
                    .trace(params, start, self.to_world(grid, cell + lower_offset))
                    .restrict_angle(lower_size),
                self.tracer
                    .trace(params, start, self.to_world(grid, cell + upper_offset))
                    .restrict_angle(upper_size),
            ]
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle<R> {
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

pub struct AnalyticTracer<F: MergeFluence> {
    pub circles: Vec<Circle<F::Radiance>>,
}

impl<F: MergeFluence> WorldTracer<F> for AnalyticTracer<F> {
    type Params = ();
    #[tracked]
    fn trace(
        &self,
        _params: &Self::Params,
        start: Expr<Vec2<f32>>,
        end: Expr<Vec2<f32>>,
    ) -> Expr<Fluence<F>> {
        let end_t = (end - start).length();
        let dir = (end - start).normalize();

        let best_t = end_t.var();
        let best_color = F::Radiance::black().var();
        for Circle {
            center,
            radius,
            color,
        } in self.circles.iter()
        {
            let (min_t, max_t, penetration, hit) =
                intersect_circle(start - center, dir, radius.expr());
            if hit && max_t > 0.0 && min_t < best_t {
                *best_t = min_t;
                // TODO: Make this adjustable / always stay at the x-axis size or something.
                // Also make it part of the opacity instead.
                *best_color = F::Radiance::scale(color.expr(), keter::min(penetration / 2.0, 1.0));
            }
        }
        if best_t < end_t {
            Fluence::solid_expr(**best_color)
        } else {
            Fluence::empty().expr()
        }
    }
}

pub struct AnalyticCursorTracer<F: MergeFluence> {
    pub circles: Vec<Circle<F::Radiance>>,
}

impl<F: MergeFluence> WorldTracer<F> for AnalyticCursorTracer<F> {
    type Params = Expr<Vec2<f32>>;
    #[tracked]
    fn trace(
        &self,
        cursor_pos: &Self::Params,
        start: Expr<Vec2<f32>>,
        end: Expr<Vec2<f32>>,
    ) -> Expr<Fluence<F>> {
        let end_t = (end - start).length();
        let dir = (end - start).normalize();

        let best_t = end_t.var();
        let best_color = F::Radiance::black().var();
        for Circle {
            center,
            radius,
            color,
        } in self.circles.iter()
        {
            let center = if *center == Vec2::splat(0.0) {
                *cursor_pos
            } else {
                center.expr()
            };
            let (min_t, max_t, penetration, hit) =
                intersect_circle(start - center, dir, radius.expr());
            if hit && max_t > 0.0 && min_t < best_t {
                *best_t = min_t;
                // TODO: Make this adjustable / always stay at the x-axis size or something.
                // Also make it part of the opacity instead.
                *best_color = F::Radiance::scale(color.expr(), keter::min(penetration / 2.0, 1.0));
            }
        }
        if best_t < end_t {
            Fluence::solid_expr(**best_color)
        } else {
            Fluence::empty().expr()
        }
    }
}

pub struct StorageTracer;
impl StorageTracer {
    #[tracked]
    pub fn load<F: MergeFluence>(
        buffer: &BufferVar<Fluence<F>>,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> Expr<Fluence<F>> {
        let cell = probe.cell.cast_u32();
        buffer.read(cell.x + grid.size.x * (cell.y + grid.size.y * probe.dir))
    }
    #[tracked]
    pub fn load_opt<F: MergeFluence>(
        buffer: &BufferVar<Fluence<F>>,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> Expr<Fluence<F>> {
        let cell = probe.cell.cast_u32();
        if (cell >= grid.size).any() {
            Fluence::empty().expr()
        } else {
            buffer.read(cell.x + grid.size.x * (cell.y + grid.size.y * probe.dir))
        }
    }
    #[tracked]
    pub fn store<F: MergeFluence>(
        buffer: &BufferVar<Fluence<F>>,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
        fluence: Expr<Fluence<F>>,
    ) {
        let cell = probe.cell.cast_u32();
        buffer.write(
            cell.x + grid.size.x * (cell.y + grid.size.y * probe.dir),
            fluence,
        );
    }
}

impl<F: MergeFluence> Tracer<F> for StorageTracer {
    type Params = (BufferVar<Fluence<F>>, BufferVar<Fluence<F>>, Expr<f32>);
    #[tracked]
    fn trace(
        &self,
        (buffer, next_buffer, spacing): &Self::Params,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> [Expr<Fluence<F>>; 2] {
        let next_grid = grid.next();
        let lower_size = next_grid.angle_size(*spacing, probe.dir * 2);
        let upper_size = next_grid.angle_size(*spacing, probe.dir * 2 + 1);
        if probe.cell.x % 2 == 0 {
            [
                StorageTracer::load(next_buffer, next_grid, probe.next().with_dir(probe.dir * 2))
                    .restrict_angle(lower_size),
                StorageTracer::load(
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
                StorageTracer::load(buffer, grid, probe).restrict_angle(lower_size),
                StorageTracer::load(buffer, grid, probe.with_dir(probe.dir + 1))
                    .restrict_angle(upper_size),
            ]
        }
    }
}

// TODO: try to prevent integer division / multiplication
#[tracked]
pub fn merge_up<F: MergeFluence>(
    grid: Expr<Grid>,
    probe: Expr<Probe>,
    last_buffer: &BufferVar<Fluence<F>>,
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
            StorageTracer::load(last_buffer, last_grid, Probe::expr(last_cell, last_dir)),
            StorageTracer::load_opt(last_buffer, last_grid, midpoint),
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
                StorageTracer::load(
                    last_buffer,
                    last_grid,
                    Probe::expr(last_cell, last_dir_lower),
                ),
                StorageTracer::load_opt(last_buffer, last_grid, midpoint_lower),
            ),
            F::over(
                StorageTracer::load(
                    last_buffer,
                    last_grid,
                    Probe::expr(last_cell, last_dir_upper),
                ),
                StorageTracer::load_opt(last_buffer, last_grid, midpoint_upper),
            ),
        )
    }
}

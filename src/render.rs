use std::marker::PhantomData;
use std::mem::swap;

use keter::graph::NodeConfigs;
use keter::lang::types::vector::Vec2;
use keter::prelude::*;

use crate::fluence::{Fluence, FluenceType, RgbF16 as F};
use crate::storage::BufferStorage;
use crate::trace::{merge_up, SegmentedWorldMapper, StorageTracer, WorldSegment, WorldTracer};
use crate::{merge_0, Axis, Grid, MergeKernel, MergeKernelSettings, Probe};

type R = <F as FluenceType>::Radiance;

pub struct HRCRenderer {
    num_cascades: u32,
    display_size: u32,
    size: u32,
    segments: Vec<WorldSegment>,
    axes: [Axis; 3],
    traced_levels: u32,
    cache_pyramid: Vec<Buffer<Fluence<F>>>,
    fluence_buffers: [Buffer<R>; 2],
    store_level_kernel: Kernel<fn(Grid, Buffer<Fluence<F>>)>,
    merge_up_kernel: Kernel<fn(Grid, Buffer<Fluence<F>>, Buffer<Fluence<F>>)>,
    kernel: MergeKernel<F, StorageTracer, BufferStorage>,
    finish_kernel: Kernel<fn(Vec2<i32>, u32, Buffer<R>)>,
    pub radiance: Tex2d<R>,
}

#[derive(Debug, Clone, Copy)]
pub struct HRCSettings {
    pub traced_levels: u32,
}

impl HRCRenderer {
    fn rotations() -> [Vec2<i32>; 4] {
        [Vec2::x(), Vec2::y(), -Vec2::x(), -Vec2::y()]
    }
    pub fn new(
        world: impl WorldTracer<F, Params = ()>,
        display_size: u32,
        settings: HRCSettings,
    ) -> Self {
        assert!(display_size.is_power_of_two());
        let size = display_size / 2;
        let num_cascades = size.trailing_zeros();

        let traced_levels = settings.traced_levels;

        let mut segments = vec![];

        for r in Self::rotations() {
            for y_offset in [0, 1] {
                let half_size = display_size as f32 / 2.0;
                let r = Vec2::new(r.x as f32, r.y as f32);
                let x_dir = r;
                let y_dir = Vec2::new(-r.y, r.x);
                let diag = x_dir + y_dir;
                segments.push(WorldSegment {
                    rotation: r,
                    origin: Vec2::splat(half_size) - half_size * diag
                        + y_dir * (y_offset as f32 + 0.499)
                        + x_dir * 0.501, // Hack to avoid "corner" cases.
                    size: Vec2::splat(half_size * 2.0),
                    offset: size * segments.len() as u32,
                });
            }
        }

        let tracer = SegmentedWorldMapper {
            tracer: &world,
            segments: DEVICE.create_buffer_from_slice(&segments),
            _marker: PhantomData::<F>,
        };
        let axes = [Axis::CellY, Axis::Direction, Axis::CellX];
        let block_size = [32, 2, 2];

        let ray_storage = StorageTracer { axes };
        let merge_storage = BufferStorage { axes };

        // TODO: Don't need to store first couple of levels.
        let cache_pyramid = (0..num_cascades + 1)
            .map(|i| {
                DEVICE.create_buffer::<Fluence<F>>(
                    ((size >> i) * size * ((2 << i) + 1)) as usize * segments.len(),
                )
            })
            .collect::<Vec<_>>();

        let fluence_buffers = [
            DEVICE.create_buffer::<R>(segments.len() * (size * size * 2) as usize),
            DEVICE.create_buffer::<R>(segments.len() * (size * size * 2) as usize),
        ];

        let store_level_kernel =
            DEVICE.create_kernel::<fn(Grid, Buffer<Fluence<F>>)>(&track!(|grid, buffer| {
                set_block_size(block_size);
                let (cell, dir) = Axis::dispatch_id(axes);
                let cell = cell.cast_i32();
                let probe = Probe::expr(cell, dir);
                let fluence = tracer.compute_ray(grid, probe, &());
                ray_storage.store(&buffer, grid, probe, fluence);
            }));

        let merge_up_kernel = DEVICE
            .create_kernel::<fn(Grid, Buffer<Fluence<F>>, Buffer<Fluence<F>>)>(&track!(
                |grid, last_buffer, buffer| {
                    set_block_size(block_size);
                    let (cell, dir) = Axis::dispatch_id(axes);
                    let cell = cell.cast_i32();
                    let probe = Probe::expr(cell, dir);
                    let fluence = merge_up(&ray_storage, &last_buffer, grid, probe);
                    ray_storage.store(&buffer, grid, probe, fluence);
                }
            ));

        let settings = MergeKernelSettings {
            axes,
            block_size,
            storage: &merge_storage,
            tracer: &ray_storage,
            _marker: PhantomData::<F>,
        };

        let kernel = settings.build_kernel();

        let radiance_texture =
            DEVICE.create_tex2d::<R>(R::natural_storage(), display_size, display_size, 1);

        let finish_kernel = DEVICE
            .create_kernel::<fn(Vec2<i32>, u32, Buffer<<F as FluenceType>::Radiance>)>(&track!(
                |rotation, rotation_index, next_radiance| {
                    set_block_size([2, 32, 2]);
                    let cell = dispatch_id().xy().cast_i32();

                    let out_cell = {
                        let r = rotation.cast_f32();
                        let half_size = display_size as f32 / 2.0;
                        let x_dir = r;
                        let y_dir = Vec2::expr(-r.y, r.x);
                        let diag = x_dir + y_dir;
                        let origin = Vec2::splat(half_size) - half_size * diag + diag * 0.5;
                        let value =
                            (origin + x_dir * cell.x.cast_f32() + y_dir * cell.y.cast_f32())
                                .floor();
                        let a = (value >= 0.0).all() && (value < display_size as f32).all();
                        lc_assert!(a);
                        value.cast_u32()
                    };

                    let cell = cell + Vec2::x();
                    if cell.x >= display_size as i32 {
                        return;
                    }

                    let next_grid =
                        Grid::new(Vec2::new(size, segments.len() as u32 * size), 2).expr();

                    let global_y_offset = (2 * size * rotation_index).cast_i32();

                    let offset = if cell.x % 2 == cell.y % 2 {
                        0.expr()
                    } else if cell.y % 2 == 0 {
                        (size - 1).expr()
                    } else {
                        size.expr()
                    };
                    let radiance = merge_0::<F, _, _>(
                        next_grid,
                        Vec2::expr(cell.x / 2, cell.y / 2 + offset.cast_i32() + global_y_offset),
                        (&merge_storage, &next_radiance),
                        (
                            &tracer,
                            // TODO: This really isn't an elegant way of solving the edge cases, but it works.
                            &(
                                2 * rotation_index + (cell.x % 2 != cell.y % 2).cast_u32(),
                                (),
                            ),
                        ),
                        cell.x % 2 == 0,
                    );

                    radiance_texture.write(out_cell, radiance_texture.read(out_cell) + radiance);
                }
            ));
        Self {
            num_cascades,
            display_size,
            size,
            segments,
            axes,
            traced_levels,
            cache_pyramid,
            fluence_buffers,
            store_level_kernel,
            merge_up_kernel,
            kernel,
            finish_kernel,
            radiance: radiance_texture,
        }
    }
    pub fn render(&self) -> NodeConfigs {
        let merge_up_commands = (0..self.num_cascades as usize + 1)
            .map(|i| {
                let grid = Grid::new(
                    Vec2::new(self.size >> i, self.segments.len() as u32 * self.size),
                    2 << i,
                );
                let dispatch_grid = Grid::new(grid.size, grid.directions + 1);
                if i < self.traced_levels as usize {
                    self.store_level_kernel.dispatch_async(
                        Axis::dispatch_size(self.axes, dispatch_grid),
                        &grid,
                        &self.cache_pyramid[i],
                    )
                } else {
                    self.merge_up_kernel.dispatch_async(
                        Axis::dispatch_size(self.axes, dispatch_grid),
                        &grid,
                        &self.cache_pyramid[i - 1],
                        &self.cache_pyramid[i],
                    )
                }
                .debug(format!("Merge Up {}", i))
            })
            .collect::<Vec<_>>()
            .chain();

        let mut buffer_a = &self.fluence_buffers[0];
        let mut buffer_b = &self.fluence_buffers[1];

        let merge_commands = (0..self.num_cascades as usize)
            .rev()
            .map(|i| {
                swap(&mut buffer_a, &mut buffer_b);
                let grid = Grid::new(
                    Vec2::new(self.size >> i, self.segments.len() as u32 * self.size),
                    2 << i,
                );
                self.kernel
                    .dispatch(
                        grid,
                        &(
                            self.cache_pyramid[i].view(..),
                            self.cache_pyramid[i + 1].view(..),
                        ),
                        buffer_b,
                        buffer_a,
                    )
                    .debug(format!("Merge {}", i))
            })
            .collect::<Vec<_>>()
            .chain();
        let finish_commands = Self::rotations()
            .iter()
            .enumerate()
            .map(|(i, r)| {
                self.finish_kernel
                    .dispatch_async(
                        [self.display_size, self.display_size, 1],
                        r,
                        &(i as u32),
                        buffer_a,
                    )
                    .debug(format!("Finish {}", i))
            })
            .collect::<Vec<_>>()
            .chain();
        (merge_up_commands, merge_commands, finish_commands).chain()
    }
}

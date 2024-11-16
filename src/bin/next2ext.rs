use std::marker::PhantomData;
use std::mem::swap;

use amitabha::color::{Fluence, MergeFluence, RgbF16};
use amitabha::storage::BufferStorage;
use amitabha::trace::{
    merge_up, AnalyticCursorTracer, Circle, SegmentedWorldMapper, StorageTracer, WorldSegment,
};
use amitabha::utils::pcgf;
use amitabha::{merge_1_even, merge_1_odd, Axis, Grid, MergeKernelSettings, Probe};
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::{App, MouseButton};

const DISPLAY_SIZE: u32 = 512;
const SIZE: u32 = DISPLAY_SIZE / 2;
const SEGMENTS: u32 = 4 * 2;

type F = RgbF16;

fn main() {
    let num_cascades = SIZE.trailing_zeros() as usize;
    let mut light_pos = Vec2::new(256.0, 256.0);

    let grid_size = [DISPLAY_SIZE; 2];
    let app = App::new("Amitabha", grid_size)
        .scale(4)
        .dpi(2.0)
        .agx()
        .init();

    let mut buffer_a = DEVICE
        .create_buffer::<<F as MergeFluence>::Radiance>((SEGMENTS * SIZE * SIZE * 2) as usize);
    let mut buffer_b = DEVICE
        .create_buffer::<<F as MergeFluence>::Radiance>((SEGMENTS * SIZE * SIZE * 2) as usize);

    let tracer = AnalyticCursorTracer::<F> {
        circles: vec![
            Circle {
                center: Vec2::new(0.0, 0.0),
                radius: 10.0,
                color: Vec3::splat(f16::from_f32_const(10.0)),
            },
            Circle {
                center: Vec2::new(200.0, 150.0),
                radius: 10.0,
                color: Vec3::splat(f16::from_f32_const(0.0)),
            },
        ],
    };

    let rotations = [Vec2::x(), Vec2::y(), -Vec2::x(), -Vec2::y()];

    let mut segments = vec![];

    for r in rotations {
        for y_offset in [0, 1] {
            segments.push(WorldSegment {
                rotation: r,
                origin: Vec2::new(256.0, 256.0) + Vec2::new(-r.y, r.x) * y_offset as f32,
                size: Vec2::new(512.0, 512.0),
                offset: SIZE * segments.len() as u32,
            });
        }
    }

    let tracer = SegmentedWorldMapper {
        tracer,
        segments: DEVICE.create_buffer_from_slice(&segments),
        _marker: PhantomData::<F>,
    };

    let store_axes = [Axis::CellY, Axis::Direction, Axis::CellX];
    let up_axes = [Axis::CellY, Axis::Direction, Axis::CellX];
    let axes = [Axis::CellY, Axis::Direction, Axis::CellX];

    let ray_storage = StorageTracer { axes };
    let merge_storage = BufferStorage { axes };

    let cache_pyramid = (0..num_cascades + 1)
        .map(|i| {
            DEVICE.create_buffer::<Fluence<F>>(
                ((SIZE >> i) * SEGMENTS * SIZE * ((2 << i) + 1)) as usize,
            )
        })
        .collect::<Vec<_>>();

    let store_level_kernel = DEVICE.create_kernel::<fn(Grid, Buffer<Fluence<F>>, Vec2<f32>)>(
        &track!(|grid, buffer, light_pos| {
            set_block_size([32, 2, 2]);
            let (cell, dir) = Axis::dispatch_id(store_axes);
            let cell = cell.cast_i32();
            let probe = Probe::expr(cell, dir);
            let fluence = tracer.compute_ray(grid, probe, &light_pos);
            ray_storage.store(&buffer, grid, probe, fluence);
        }),
    );

    let merge_up_kernel = DEVICE.create_kernel::<fn(Grid, Buffer<Fluence<F>>, Buffer<Fluence<F>>)>(
        &track!(|grid, last_buffer, buffer| {
            set_block_size([32, 2, 2]);
            let (cell, dir) = Axis::dispatch_id(up_axes);
            let cell = cell.cast_i32();
            let probe = Probe::expr(cell, dir);
            let fluence = merge_up(&ray_storage, &last_buffer, grid, probe);
            ray_storage.store(&buffer, grid, probe, fluence);
        }),
    );

    let settings = MergeKernelSettings {
        axes,
        block_size: [32, 2, 2],
        storage: &merge_storage,
        tracer: &ray_storage,
        _marker: PhantomData::<F>,
    };

    let kernel = settings.build_kernel();

    let draw = DEVICE.create_kernel::<fn(Vec2<f32>, u32, Buffer<<F as MergeFluence>::Radiance>)>(
        &track!(|rotation, rotation_index, next_radiance| {
            let cell = dispatch_id().xy();
            let cell = cell.cast_f32() - Vec2::splat(DISPLAY_SIZE as f32 / 2.0);
            let cell = Vec2::expr(
                cell.x * rotation.x - cell.y * rotation.y,
                cell.x * rotation.y + cell.y * rotation.x,
            );
            let cell = cell + Vec2::splat(DISPLAY_SIZE as f32 / 2.0);

            let cell = cell + Vec2::x();
            let cell = cell.round().cast_i32();

            let grid = Grid::new(Vec2::new(DISPLAY_SIZE, SEGMENTS * SIZE), 1).expr();

            // TODO: Divide by 2 is bad.
            let radiance = if cell.x % 2 == 0 {
                let offset = if cell.y % 2 == 0 {
                    (2 * SIZE * rotation_index).cast_i32()
                } else {
                    (SIZE + 2 * SIZE * rotation_index).cast_i32()
                };
                merge_1_even::<F, _>(
                    grid,
                    Vec2::expr(cell.x + 1, cell.y / 2 + offset),
                    (&merge_storage, &next_radiance),
                )
            } else {
                let offset = if cell.y % 2 == 0 {
                    -1 + (SIZE + 2 * SIZE * rotation_index).cast_i32()
                } else {
                    (2 * SIZE * rotation_index).cast_i32()
                };
                // Need to collect from odd cells.
                merge_1_odd::<F, _>(
                    grid,
                    Vec2::expr(cell.x + 1, cell.y / 2 + offset),
                    (&merge_storage, &next_radiance),
                )
            };

            app.display().write(
                dispatch_id().xy(),
                app.display().read(dispatch_id().xy()) + radiance.cast_f32(),
            );
        }),
    );

    let apply_noise = DEVICE.create_kernel::<fn()>(&track!(|| {
        let noise = pcgf(dispatch_id().x + (dispatch_id().y << 16));
        let noise = noise / 255.0 * 6.0;
        app.display().write(
            dispatch_id().xy(),
            app.display().read(dispatch_id().xy()) + Vec3::splat_expr(noise),
        );
    }));

    let mut merge_timings = vec![vec![]; num_cascades];
    let mut merge_up_timings = vec![vec![]; num_cascades + 1];

    app.run(|rt, _scope| {
        if rt.pressed_button(MouseButton::Left) {
            light_pos = rt.cursor_position;
        }
        let merge_up_commands = (0..num_cascades + 1)
            .map(|i| {
                let grid = Grid::new(Vec2::new(SIZE >> i, SEGMENTS * SIZE), 2 << i);
                let dispatch_grid = Grid::new(grid.size, grid.directions + 1);
                if i < 2 {
                    store_level_kernel.dispatch_async(
                        Axis::dispatch_size(store_axes, dispatch_grid),
                        &grid,
                        &cache_pyramid[i],
                        &light_pos,
                    )
                } else {
                    merge_up_kernel.dispatch_async(
                        Axis::dispatch_size(up_axes, dispatch_grid),
                        &grid,
                        &cache_pyramid[i - 1],
                        &cache_pyramid[i],
                    )
                }
                .debug(format!("merge-up-{}", i))
            })
            .collect::<Vec<_>>()
            .chain();
        let timings = merge_up_commands.execute_timed();
        for (name, time) in timings.iter() {
            let i = name.split('-').last().unwrap().parse::<usize>().unwrap();
            merge_up_timings[i].push(*time as f64);
        }

        let merge_commands = (0..num_cascades)
            .rev()
            .map(|i| {
                swap(&mut buffer_a, &mut buffer_b);
                let grid = Grid::new(Vec2::new(SIZE >> i, SEGMENTS * SIZE), 2 << i);
                kernel
                    .dispatch(
                        grid,
                        &(
                            cache_pyramid[i].view(..),
                            cache_pyramid[i + 1].view(..),
                            (1 << (i + 1)) as f32,
                        ),
                        &buffer_a,
                        &buffer_b,
                    )
                    .debug(format!("merge-{}", i))
            })
            .collect::<Vec<_>>()
            .chain();
        let timings = merge_commands.execute_timed();
        for (name, time) in timings.iter() {
            let i = name.split('-').last().unwrap().parse::<usize>().unwrap();
            merge_timings[i].push(*time as f64);
        }

        for (i, r) in rotations.iter().enumerate() {
            draw.dispatch(
                [DISPLAY_SIZE, DISPLAY_SIZE, 1],
                &Vec2::new(r.x, -r.y),
                &(i as u32),
                &buffer_a,
            );
        }
        // apply_noise.dispatch([DISPLAY_SIZE, DISPLAY_SIZE, 1]);

        if rt.tick % 200 == 0 {
            let c = 200.0;
            println!("Merge Timings:");
            let mut total = 0.0;
            let mut total_variance = 0.0;
            for (i, timings) in merge_timings.iter_mut().enumerate() {
                let avg = timings.iter().sum::<f64>() / c;
                total += avg;
                let variance = timings.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / c;
                total_variance += variance;
                // println!("    {}: {:.2}ms, sd {:.3}ms", i, avg, variance.sqrt());
                timings.clear();
            }
            println!("Total: {:.3}ms, sd {:.3}ms", total, total_variance.sqrt());

            println!("Merge Up Timings:");
            let mut total = 0.0;
            let mut total_variance = 0.0;
            for (i, timings) in merge_up_timings.iter_mut().enumerate() {
                let avg = timings.iter().sum::<f64>() / c;
                total += avg;
                let variance = timings.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / c;
                total_variance += variance;
                // println!("    {}: {:.2}ms, sd {:.3}ms", i, avg, variance.sqrt());
                timings.clear();
            }
            println!("Total: {:.3}ms, sd {:.3}ms", total, total_variance.sqrt());
        }
    });
}

use std::marker::PhantomData;
use std::mem::swap;
use std::time::Instant;

use amitabha::color::{Fluence, SingleF32};
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

fn main() {
    let num_cascades = SIZE.trailing_zeros() as usize;
    let mut light_pos = Vec2::new(256.0, 256.0);

    let grid_size = [DISPLAY_SIZE; 2];
    let app = App::new("Amitabha", grid_size)
        .scale(4)
        .dpi(2.0)
        .agx()
        .init();

    let mut buffer_a = DEVICE.create_buffer::<f32>((SEGMENTS * SIZE * SIZE * 2) as usize);
    let mut buffer_b = DEVICE.create_buffer::<f32>((SEGMENTS * SIZE * SIZE * 2) as usize);

    let tracer = AnalyticCursorTracer::<SingleF32> {
        circles: vec![
            Circle {
                center: Vec2::new(0.0, 0.0),
                radius: 10.0,
                color: 10.0,
            },
            Circle {
                center: Vec2::new(200.0, 150.0),
                radius: 10.0,
                color: 0.0,
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
        _marker: PhantomData::<SingleF32>,
    };

    let axes = [Axis::CellY, Axis::Direction, Axis::CellX];

    let ray_storage = StorageTracer { axes };
    let merge_storage = BufferStorage { axes };

    let cache_pyramid = (0..num_cascades + 1)
        .map(|i| {
            DEVICE.create_buffer::<Fluence<SingleF32>>(
                ((SIZE >> i) * SEGMENTS * SIZE * ((2 << i) + 1)) as usize,
            )
        })
        .collect::<Vec<_>>();

    let merge_up_kernel = DEVICE
        .create_kernel::<fn(Grid, Buffer<Fluence<SingleF32>>, Buffer<Fluence<SingleF32>>)>(
            &track!(|grid, last_buffer, buffer| {
                let cell = dispatch_id().xy().cast_i32();
                let dir = dispatch_id().z;
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
        _marker: PhantomData::<SingleF32>,
    };

    let kernel = settings.build_kernel();

    let draw = DEVICE.create_kernel::<fn(Vec2<f32>, u32, Buffer<f32>)>(&track!(
        |rotation, rotation_index, next_radiance| {
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
                if cell.y % 2 == 0 {
                    merge_1_even::<SingleF32, _>(
                        grid,
                        Vec2::expr(
                            cell.x + 1,
                            cell.y / 2 + (2 * SIZE * rotation_index).cast_i32(),
                        ),
                        (&merge_storage, &next_radiance),
                    )
                } else {
                    merge_1_even::<SingleF32, _>(
                        grid,
                        Vec2::expr(
                            cell.x + 1,
                            cell.y / 2 + (SIZE + 2 * SIZE * rotation_index).cast_i32(),
                        ),
                        (&merge_storage, &next_radiance),
                    )
                }
            } else {
                if cell.y % 2 == 0 {
                    // Need to collect from odd cells.
                    merge_1_odd::<SingleF32, _>(
                        grid,
                        Vec2::expr(
                            cell.x + 1,
                            cell.y / 2 - 1 + (SIZE + 2 * SIZE * rotation_index).cast_i32(),
                        ),
                        (&merge_storage, &next_radiance),
                    )
                } else {
                    merge_1_odd::<SingleF32, _>(
                        grid,
                        Vec2::expr(
                            cell.x + 1,
                            cell.y / 2 + (2 * SIZE * rotation_index).cast_i32(),
                        ),
                        (&merge_storage, &next_radiance),
                    )
                }
            };

            app.display().write(
                dispatch_id().xy(),
                app.display().read(dispatch_id().xy()) + Vec3::splat_expr(radiance),
            );
        }
    ));

    let apply_noise = DEVICE.create_kernel::<fn()>(&track!(|| {
        let noise = pcgf(dispatch_id().x + (dispatch_id().y << 16));
        let noise = noise / 255.0 * 6.0;
        app.display().write(
            dispatch_id().xy(),
            app.display().read(dispatch_id().xy()) + Vec3::splat_expr(noise),
        );
    }));

    let mut merge_timings = vec![vec![]; num_cascades];

    app.run(|rt, _scope| {
        if rt.pressed_button(MouseButton::Left) {
            light_pos = rt.cursor_position;
        }
        for i in 0..num_cascades + 1 {
            let grid = Grid::new(Vec2::new(SIZE >> i, SEGMENTS * SIZE), 2 << i);
            if i < 2 {
                tracer.cache_level(grid, &ray_storage, &cache_pyramid[i], &light_pos);
            } else {
                merge_up_kernel.dispatch(
                    [grid.size.x, grid.size.y, grid.directions + 1],
                    &grid,
                    &cache_pyramid[i - 1],
                    &cache_pyramid[i],
                );
            }
        }

        let start = Instant::now();

        for i in (0..num_cascades).rev() {
            swap(&mut buffer_a, &mut buffer_b);
            let grid = Grid::new(Vec2::new(SIZE >> i, SEGMENTS * SIZE), 2 << i);
            let timings = kernel
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
                .execute_timed();
            let time = timings.iter().map(|(_, t)| *t as f64).sum::<f64>();
            merge_timings[i].push(time);
        }

        let elapsed = start.elapsed();
        if rt.tick % 100 == 0 {
            println!("Elapsed: {:?}", elapsed);
        }

        for (i, r) in rotations.iter().enumerate() {
            draw.dispatch(
                [DISPLAY_SIZE, DISPLAY_SIZE, 1],
                &Vec2::new(r.x, -r.y),
                &(i as u32),
                &buffer_a,
            );
        }
        apply_noise.dispatch([DISPLAY_SIZE, DISPLAY_SIZE, 1]);

        if rt.tick % 100 == 0 {
            println!("Merge Timings:");
            let mut total = 0.0;
            let mut total_variance = 0.0;
            for (i, timings) in merge_timings.iter_mut().enumerate() {
                let avg = timings.iter().sum::<f64>() / 100.0;
                total += avg;
                let variance = timings.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / 100.0;
                total_variance += variance;
                println!("    {}: {:.2}ms, sd {:.3}ms", i, avg, variance.sqrt());
                timings.clear();
            }
            println!("Total: {:.3}ms, sd {:.3}ms", total, total_variance.sqrt());
        }
    });
}

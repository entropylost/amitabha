#![feature(more_float_constants)]

use std::f32::consts::{PHI, PI, TAU};
use std::marker::PhantomData;
use std::mem::swap;

use amitabha::color::{Color, Emission};
use amitabha::fluence::{self, Fluence, FluenceType};
use amitabha::storage::{BufferStorage, RadianceStorage};
use amitabha::trace::{
    merge_up, SegmentedWorldMapper, StorageTracer, VoxelTracer, WorldSegment, WorldTracer,
};
use amitabha::utils::gaussian;
use amitabha::{color, merge_0, Axis, Grid, MergeKernelSettings, Probe};
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::{App, KeyCode, MouseButton};

const DISPLAY_SIZE: u32 = 512;
const SIZE: u32 = DISPLAY_SIZE / 2;
const SEGMENTS: u32 = 4 * 2;

type F = fluence::RgbF16;
type R = <F as FluenceType>::Radiance;
type C = color::RgbF16;

fn main() {
    let num_cascades = SIZE.trailing_zeros() as usize;

    let grid_size = [DISPLAY_SIZE; 2];
    let app = App::new("Amitabha", grid_size)
        .scale(2048 / DISPLAY_SIZE)
        .dpi(2.0)
        .agx()
        .init();

    let mut buffer_a =
        DEVICE.create_buffer::<<F as FluenceType>::Radiance>((SEGMENTS * SIZE * SIZE * 2) as usize);
    let mut buffer_b =
        DEVICE.create_buffer::<<F as FluenceType>::Radiance>((SEGMENTS * SIZE * SIZE * 2) as usize);

    let world_size = Vec2::new(DISPLAY_SIZE, DISPLAY_SIZE);
    let tracer = VoxelTracer::<C>::new(world_size);
    let world = tracer.buffer.view(..);

    let rotations: [Vec2<i32>; 4] = [Vec2::x(), Vec2::y(), -Vec2::x(), -Vec2::y()];

    let mut segments = vec![];

    for r in rotations {
        for y_offset in [0, 1] {
            let half_size = world_size.x as f32 / 2.0;
            let r = Vec2::new(r.x as f32, r.y as f32);
            let x_dir = r;
            let y_dir = Vec2::new(-r.y, r.x);
            let diag = x_dir + y_dir;
            segments.push(WorldSegment {
                rotation: r,
                origin: Vec2::splat(half_size) - half_size * diag
                    + y_dir * (y_offset as f32 + 0.5)
                    + x_dir * 0.5,
                size: Vec2::splat(half_size * 2.0),
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

    let store_level_kernel =
        DEVICE.create_kernel::<fn(Grid, Buffer<Fluence<F>>)>(&track!(|grid, buffer| {
            set_block_size([32, 2, 2]);
            let (cell, dir) = Axis::dispatch_id(store_axes);
            let cell = cell.cast_i32();
            let probe = Probe::expr(cell, dir);
            let fluence = tracer.compute_ray(grid, probe, &());
            ray_storage.store(&buffer, grid, probe, fluence);
        }));

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

    let radiance_texture =
        DEVICE.create_tex2d::<Vec3<f32>>(PixelStorage::Float4, DISPLAY_SIZE, DISPLAY_SIZE, 1);

    let draw = DEVICE.create_kernel::<fn(Vec2<i32>, u32, Buffer<<F as FluenceType>::Radiance>)>(
        &track!(|rotation, rotation_index, next_radiance| {
            let cell = dispatch_id().xy().cast_i32();

            let out_cell = {
                let r = rotation.cast_f32();
                let half_size = world_size.x as f32 / 2.0;
                let x_dir = r;
                let y_dir = Vec2::expr(-r.y, r.x);
                let diag = x_dir + y_dir;
                let origin = Vec2::splat(half_size) - half_size * diag + diag * 0.5;
                let value =
                    (origin + x_dir * cell.x.cast_f32() + y_dir * cell.y.cast_f32()).floor();
                let a = (value >= 0.0).all() && (value < DISPLAY_SIZE as f32).all();
                lc_assert!(a);
                value.cast_u32()
            };

            let cell = cell + Vec2::x();
            if cell.x >= DISPLAY_SIZE as i32 {
                return;
            }

            let next_grid = Grid::new(Vec2::new(SIZE, SEGMENTS * SIZE), 2).expr();

            let global_y_offset = (2 * SIZE * rotation_index).cast_i32();

            let offset = if cell.x % 2 == cell.y % 2 {
                0.expr()
            } else if cell.y % 2 == 0 {
                (SIZE - 1).expr()
            } else {
                SIZE.expr()
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

            // This is still necessary, since the even-line load still is offset by 1.
            let base_fluence = world
                .read(out_cell.x + out_cell.y * world_size.x)
                .to_fluence::<F>(0.5.expr())
                .restrict_angle((PI / 2.0).expr());

            let radiance = base_fluence.over_radiance(radiance);

            radiance_texture.write(
                out_cell,
                radiance_texture.read(out_cell) + radiance.cast_f32(),
            );
        }),
    );

    let blur = 0.25;
    let delta = 0.05;

    let filter = DEVICE.create_kernel::<fn()>(&track!(|| {
        let pos = dispatch_id().xy();
        let value = radiance_texture.read(pos);
        let denom = 0.0.var();
        let numer = Vec3::splat(0.0_f32).var();
        if (world.read(pos.x + pos.y * world_size.x).opacity != f16::ZERO).any() {
            app.display().write(pos, value);
            return;
        }
        for (offset, weight) in [
            (Vec2::<i32>::splat(0), 1.0),
            (Vec2::x(), blur),
            (Vec2::y(), blur),
            (-Vec2::x(), blur),
            (-Vec2::y(), blur),
        ] {
            let pos = (pos.cast_i32() + offset).cast_u32();
            // TODO: Bias surfaces as well, and do more general thing.
            // Can difference this and other opacity / optical depth?
            if (pos < DISPLAY_SIZE).all()
                && (world.read(pos.x + pos.y * world_size.x).opacity == f16::ZERO).any()
            {
                let neighbor = radiance_texture.read(pos);
                let weight = weight * gaussian((neighbor - value).reduce_max() / delta);
                *numer += neighbor * weight;
                *denom += weight;
            }
        }
        app.display().write(pos, numer / denom);
    }));
    let reset_texture = DEVICE.create_kernel::<fn(Tex2d<Vec3<f32>>)>(&track!(|texture| {
        texture.write(dispatch_id().xy(), Vec3::splat(0.0));
    }));

    let pt_texture =
        DEVICE.create_tex2d::<Vec3<f32>>(PixelStorage::Float4, DISPLAY_SIZE, DISPLAY_SIZE, 1);
    let mut pt_count = 0;
    let mut display_pt = false;
    let path_trace = DEVICE.create_kernel::<fn(u32, u32)>(&track!(|t, n| {
        let world_pos = dispatch_id().xy().cast_f32() + Vec2::splat(0.5);
        let total_radiance = Vec3::splat(0.0_f32).var();
        for i in 0.expr()..n {
            let dir = ((PHI * (t + i).cast_f32()) % 1.0) * TAU;
            let dir = Vec2::expr(dir.cos(), dir.sin());
            let radiance = tracer
                .tracer
                .trace(&(), world_pos, dir, (2.0 * DISPLAY_SIZE as f32).expr())
                .radiance;
            *total_radiance += radiance.cast_f32();
        }
        pt_texture.write(
            dispatch_id().xy(),
            pt_texture.read(dispatch_id().xy()) + total_radiance,
        );
    }));
    let draw_pt = DEVICE.create_kernel::<fn(u32)>(&track!(|pt_count| {
        let pos = dispatch_id().xy();
        let radiance = pt_texture.read(pos) / pt_count.cast_f32();
        app.display().write(pos, app.display().read(pos) + radiance);
    }));

    let draw_circle =
        DEVICE.create_kernel::<fn(Vec2<f32>, f32, Color<C>)>(&track!(|center, radius, color| {
            let pos = dispatch_id().xy();
            if (pos.cast_f32() - center).abs().reduce_max() < radius {
                world.write(pos.x + pos.y * world_size.x, color);
            }
        }));

    let draw_solid = DEVICE.create_kernel::<fn()>(&track!(|| {
        let pos = dispatch_id().xy();
        if (world.read(pos.x + pos.y * world_size.x).opacity != f16::ZERO).any() {
            app.display().write(pos, Vec3::expr(1.0, 0.0, 0.0));
        }
    }));

    let blur = DEVICE.create_kernel::<fn(Grid, f32, Buffer<R>, Buffer<R>)>(&track!(
        |grid, blur, buffer_x, buffer_y| {
            let x = dispatch_id().x.cast_i32();
            let y = dispatch_id().y;
            let segment = 2 * (y / DISPLAY_SIZE);
            let y = (y % DISPLAY_SIZE).cast_i32();
            let dir = dispatch_id().z;

            let get_index = |y: Expr<i32>| {
                let y = y.clamp(0_i32, DISPLAY_SIZE as i32);
                let y = if y % 2 == 0 {
                    y / 2
                } else {
                    y / 2 + SIZE as i32
                };
                Probe::expr(Vec2::expr(x, y + (segment * SIZE).cast_i32()), dir)
            };

            let denom = 2.0 * blur + 1.0;
            let numer = merge_storage.load(&buffer_x, grid, get_index(y))
                + blur.cast_f16()
                    * (merge_storage.load(&buffer_a, grid, get_index(y - 1))
                        + merge_storage.load(&buffer_a, grid, get_index(y + 1)));
            merge_storage.store(&buffer_y, grid, get_index(y), numer / denom.cast_f16());
        }
    ));

    let mut display_solid = false;

    let mut merge_timings = vec![vec![]; num_cascades];
    let mut merge_up_timings = vec![vec![]; num_cascades + 1];

    app.run(|rt, _scope| {
        if rt.pressed_button(MouseButton::Middle) {
            draw_circle.dispatch(
                [world_size.x, world_size.y, 1],
                &rt.cursor_position,
                &4.0,
                &Color::solid(Vec3::new(f16::ONE, f16::ZERO, f16::ZERO)),
            );
        }
        if rt.pressed_button(MouseButton::Left) {
            draw_circle.dispatch(
                [world_size.x, world_size.y, 1],
                &rt.cursor_position,
                &4.0,
                &Color::solid(Vec3::black()),
            );
        }
        if rt.pressed_button(MouseButton::Right) {
            draw_circle.dispatch(
                [world_size.x, world_size.y, 1],
                &rt.cursor_position,
                &4.0,
                &Color::empty(),
            );
        }
        if rt.pressed_button(MouseButton::Back) {
            draw_circle.dispatch(
                [world_size.x, world_size.y, 1],
                &rt.cursor_position,
                &4.0,
                &Color::solid(Vec3::new(f16::ZERO, f16::ONE, f16::ONE)),
            );
        }

        let merge_up_commands = (0..num_cascades + 1)
            .map(|i| {
                let grid = Grid::new(Vec2::new(SIZE >> i, SEGMENTS * SIZE), 2 << i);
                let dispatch_grid = Grid::new(grid.size, grid.directions + 1);
                if i < 4 {
                    store_level_kernel.dispatch_async(
                        Axis::dispatch_size(store_axes, dispatch_grid),
                        &grid,
                        &cache_pyramid[i],
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
                // swap(&mut buffer_a, &mut buffer_b);
                let grid = Grid::new(Vec2::new(SIZE >> i, SEGMENTS * SIZE), 2 << i);
                (
                    kernel
                        .dispatch(
                            grid,
                            &(
                                cache_pyramid[i].view(..),
                                cache_pyramid[i + 1].view(..),
                                (1 << (i + 1)) as f32,
                            ),
                            &buffer_b,
                            &buffer_a,
                        )
                        .debug(format!("merge-{}", i)),
                    blur.dispatch_async(
                        [SIZE >> i, SEGMENTS * SIZE, 2 << i],
                        &grid,
                        &0.1,
                        &buffer_b,
                        &buffer_a,
                    )
                    .debug("blur"),
                )
                    .chain()
            })
            .collect::<Vec<_>>()
            .chain();
        let timings = merge_commands.execute_timed();
        for (name, time) in timings.iter() {
            if name != "blur" {
                let i = name.split('-').last().unwrap().parse::<usize>().unwrap();
                merge_timings[i].push(*time as f64);
            }
        }

        for (i, r) in rotations.iter().enumerate() {
            draw.dispatch([DISPLAY_SIZE, DISPLAY_SIZE, 1], r, &(i as u32), &buffer_a);
        }

        if rt.just_pressed_key(KeyCode::KeyP) {
            display_pt = !display_pt;
            pt_count = 0;
            reset_texture.dispatch([DISPLAY_SIZE, DISPLAY_SIZE, 1], &pt_texture);
        }
        if display_pt {
            let n = 20;
            path_trace.dispatch([DISPLAY_SIZE, DISPLAY_SIZE, 1], &pt_count, &n);
            draw_pt.dispatch([DISPLAY_SIZE, DISPLAY_SIZE, 1], &pt_count);
            pt_count += n;
        } else {
            filter.dispatch([DISPLAY_SIZE, DISPLAY_SIZE, 1]);
        }
        reset_texture.dispatch([DISPLAY_SIZE, DISPLAY_SIZE, 1], &radiance_texture);

        if rt.just_pressed_key(KeyCode::KeyB) {
            display_solid = !display_solid;
        }
        if display_solid {
            draw_solid.dispatch([DISPLAY_SIZE, DISPLAY_SIZE, 1]);
        }

        if rt.tick % 1000 == 0 {
            let enumerate = false;

            let c = 1000.0;
            println!("Merge Timings:");
            let mut total = 0.0;
            let mut total_variance = 0.0;
            for (i, timings) in merge_timings.iter_mut().enumerate() {
                let avg = timings.iter().sum::<f64>() / c;
                total += avg;
                let variance = timings.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / c;
                total_variance += variance;
                if enumerate {
                    println!("    {}: {:.2}ms, sd {:.3}ms", i, avg, variance.sqrt());
                }
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
                if enumerate {
                    println!("    {}: {:.2}ms, sd {:.3}ms", i, avg, variance.sqrt());
                }
                timings.clear();
            }
            println!("Total: {:.3}ms, sd {:.3}ms", total, total_variance.sqrt());
        }
    });
}

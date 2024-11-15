use std::marker::PhantomData;
use std::mem::swap;

use amitabha::color::SingleF32;
use amitabha::storage::BufferStorage;
use amitabha::trace::{AnalyticCursorTracer, Circle, SegmentedFrustrumWorldMapper, WorldSegment};
use amitabha::{merge_1_even, merge_1_odd, DispatchAxis, Grid, MergeKernelSettings};
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::{App, MouseButton};

const DISPLAY_SIZE: u32 = 512;
const SIZE: u32 = DISPLAY_SIZE / 2;
const SEGMENTS: u32 = 4 * 2;

fn main() {
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

    let tracer = SegmentedFrustrumWorldMapper {
        tracer,
        segments: DEVICE.create_buffer_from_slice(&segments),
        frustrums: 10,
        _marker: PhantomData::<SingleF32>,
    };

    let settings = MergeKernelSettings {
        dir_axis: DispatchAxis::Z,
        cell_axis: [DispatchAxis::X, DispatchAxis::Y],
        block_size: [8, 8, 1],
        storage: &BufferStorage,
        tracer: &tracer,
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

            let radiance = if cell.x % 2 == 0 {
                if cell.y % 2 == 0 {
                    merge_1_even::<SingleF32, _>(
                        grid,
                        Vec2::expr(
                            cell.x + 1,
                            cell.y / 2 + (2 * SIZE * rotation_index).cast_i32(),
                        ),
                        (&BufferStorage, &next_radiance),
                    )
                } else {
                    merge_1_even::<SingleF32, _>(
                        grid,
                        Vec2::expr(
                            cell.x + 1,
                            cell.y / 2 + (SIZE + 2 * SIZE * rotation_index).cast_i32(),
                        ),
                        (&BufferStorage, &next_radiance),
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
                        (&BufferStorage, &next_radiance),
                    )
                } else {
                    merge_1_odd::<SingleF32, _>(
                        grid,
                        Vec2::expr(
                            cell.x + 1,
                            cell.y / 2 + (2 * SIZE * rotation_index).cast_i32(),
                        ),
                        (&BufferStorage, &next_radiance),
                    )
                }
            };

            app.display().write(
                dispatch_id().xy(),
                app.display().read(dispatch_id().xy()) + Vec3::splat_expr(radiance),
            );
        }
    ));

    let num_cascades = SIZE.trailing_zeros() as usize;

    let mut light_pos = Vec2::new(256.0, 256.0);

    app.run(|rt, _scope| {
        if rt.pressed_button(MouseButton::Left) {
            light_pos = rt.cursor_position;
        }
        for i in (0..num_cascades).rev() {
            swap(&mut buffer_a, &mut buffer_b);
            let grid = Grid::new(Vec2::new(SIZE >> i, SEGMENTS * SIZE), 2 << i);
            kernel
                .dispatch(grid, &light_pos, &buffer_a, &buffer_b)
                .execute();
        }
        for (i, r) in rotations.iter().enumerate() {
            draw.dispatch(
                [DISPLAY_SIZE, DISPLAY_SIZE, 1],
                &Vec2::new(r.x, -r.y),
                &(i as u32),
                &buffer_a,
            );
        }

        rt.log_fps();
    });
}

use std::marker::PhantomData;
use std::mem::swap;

use amitabha::color::BinaryF32;
use amitabha::storage::{BufferStorage, RadianceStorage};
use amitabha::trace::{AnalyticCursorTracer, Circle, WorldMapper};
use amitabha::{DispatchAxis, Grid, MergeKernelSettings, Probe};
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::{App, MouseButton};

const DISPLAY_SIZE: u32 = 512;
const SIZE: u32 = DISPLAY_SIZE / 2;

fn main() {
    let grid_size = [DISPLAY_SIZE; 2];
    let app = App::new("Amitabha", grid_size)
        .scale(4)
        .dpi(2.0)
        .agx()
        .init();

    let mut buffer_a = DEVICE.create_buffer::<f32>((SIZE * SIZE * 2) as usize);
    let mut buffer_b = DEVICE.create_buffer::<f32>((SIZE * SIZE * 2) as usize);

    let rotations = [Vec2::x(), Vec2::y(), -Vec2::x(), -Vec2::y()];

    let kernels = rotations.map(|rotation| {
        let tracer = WorldMapper {
            tracer: AnalyticCursorTracer::<BinaryF32> {
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
            },
            rotation,
            world_origin: Vec2::splat(256.0),
            world_size: Vec2::splat(512.0),
            _marker: PhantomData::<BinaryF32>,
        };

        let settings = MergeKernelSettings {
            dir_axis: DispatchAxis::Z,
            cell_axis: [DispatchAxis::X, DispatchAxis::Y],
            block_size: [8, 8, 1],
            storage: &BufferStorage,
            tracer: &tracer,
            _marker: PhantomData::<BinaryF32>,
        };

        settings.build_kernel()
    });

    // Coordinates of these points in the world are at (2x + 1, 2y - 2).
    let final_buffer = DEVICE.create_buffer::<f32>((SIZE * SIZE) as usize);

    let finish = DEVICE.create_kernel::<fn(Buffer<f32>)>(&track!(|buffer| {
        let cell = dispatch_id().xy();
        let radiance_lower = BufferStorage.load(
            &buffer,
            Grid::new(Vec2::new(SIZE, SIZE), 2).expr(),
            Probe::expr(cell, 0_u32.expr()),
        );
        let radiance_upper = BufferStorage.load(
            &buffer,
            Grid::new(Vec2::new(SIZE, SIZE), 2).expr(),
            Probe::expr(cell + Vec2::y(), 1_u32.expr()),
        );
        let radiance = radiance_lower + radiance_upper;
        final_buffer.write(cell.x + cell.y * SIZE, radiance);
    }));

    let draw = DEVICE.create_kernel::<fn(Vec2<f32>)>(&track!(|rotation| {
        let cell = dispatch_id().xy();
        let cell = cell.cast_f32() - Vec2::splat(DISPLAY_SIZE as f32 / 2.0);
        let cell = Vec2::expr(
            cell.x * rotation.x - cell.y * rotation.y,
            cell.x * rotation.y + cell.y * rotation.x,
        );
        let cell = cell + Vec2::splat(DISPLAY_SIZE as f32 / 2.0);
        let cell = cell.round();
        let cell = cell - Vec2::new(-2.0, 1.0);
        let cell = cell / 2.0;

        let radiance = 0.0_f32.var();
        for c in [
            Vec2::expr(cell.x.floor(), cell.y.floor()),
            Vec2::expr(cell.x.ceil(), cell.y.floor()),
            Vec2::expr(cell.x.floor(), cell.y.ceil()),
            Vec2::expr(cell.x.ceil(), cell.y.ceil()),
        ] {
            if (c >= 0.0).all() && (c < SIZE as f32).all() {
                *radiance += final_buffer.read(c.x.cast_u32() + c.y.cast_u32() * SIZE);
            }
        }

        app.display().write(
            dispatch_id().xy(),
            app.display().read(dispatch_id().xy()) + Vec3::splat_expr(radiance / 4.0),
        );
    }));

    let num_cascades = SIZE.trailing_zeros() as usize;

    let mut light_pos = Vec2::new(256.0, 256.0);

    app.run(|rt, _scope| {
        if rt.pressed_button(MouseButton::Left) {
            light_pos = rt.cursor_position;
        }
        for r in 0..4 {
            for i in (0..num_cascades).rev() {
                swap(&mut buffer_a, &mut buffer_b);
                let grid = Grid::new(Vec2::new(SIZE >> i, SIZE), 2 << i);
                kernels[r]
                    .dispatch(grid, &light_pos, &buffer_a, &buffer_b)
                    .execute();
            }
            finish.dispatch([SIZE, SIZE, 1], &buffer_a);
            draw.dispatch(
                [DISPLAY_SIZE, DISPLAY_SIZE, 1],
                &Vec2::new(rotations[r].x, -rotations[r].y),
            );
        }

        rt.log_fps();
    });
}

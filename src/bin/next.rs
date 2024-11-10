use std::marker::PhantomData;
use std::mem::swap;

use amitabha::color::BinaryF32;
use amitabha::storage::{BufferStorage, RadianceStorage};
use amitabha::trace::{AnalyticTracer, Circle, WorldMapper};
use amitabha::{DispatchAxis, Grid, MergeKernelSettings, Probe};
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::App;

const SIZE: u32 = 512;

fn main() {
    let grid_size = [SIZE; 2];
    let app = App::new("Amitabha", grid_size)
        .scale(4)
        .dpi(2.0)
        .agx()
        .init();

    let mut buffer_a = DEVICE.create_buffer::<f32>((SIZE * SIZE) as usize);
    let mut buffer_b = DEVICE.create_buffer::<f32>((SIZE * SIZE) as usize);

    let rotations = [Vec2::x(), Vec2::y(), -Vec2::x(), -Vec2::y()];

    let kernels = rotations.map(|rotation| {
        let tracer = WorldMapper {
            tracer: AnalyticTracer::<BinaryF32> {
                circles: vec![
                    Circle {
                        center: Vec2::new(256.0, 256.0),
                        radius: 4.0,
                        color: 50.0,
                    },
                    Circle {
                        center: Vec2::new(200.0, 150.0),
                        radius: 4.0,
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

    let draw = DEVICE.create_kernel::<fn(Vec2<f32>, Buffer<f32>)>(&track!(|rotation, buffer| {
        let cell = dispatch_id().xy();
        let cell = cell.cast_f32() - Vec2::splat(SIZE as f32 / 2.0);
        let cell = Vec2::expr(
            cell.x * rotation.x - cell.y * rotation.y,
            cell.x * rotation.y + cell.y * rotation.x,
        );
        let cell = cell + Vec2::splat(SIZE as f32 / 2.0);
        let cell = cell + Vec2::y();
        let cell = cell.round().cast_u32();
        let radiance = BufferStorage.load(
            &buffer,
            Grid::new(Vec2::from(grid_size), 1).expr(),
            Probe::expr(cell, 0_u32.expr()),
        );
        app.display().write(
            dispatch_id().xy(),
            app.display().read(dispatch_id().xy()) + Vec3::splat_expr(radiance),
        );
    }));

    let num_cascades = SIZE.trailing_zeros() as usize;

    app.run(|rt, _scope| {
        for r in 0..4 {
            for i in (0..num_cascades).rev() {
                swap(&mut buffer_a, &mut buffer_b);
                let grid = Grid::new(Vec2::new(SIZE, SIZE >> i), 1 << i);
                kernels[r]
                    .dispatch(grid, &rt.cursor_position, &buffer_a, &buffer_b)
                    .execute();
            }
            draw.dispatch(
                [SIZE, SIZE, 1],
                &Vec2::new(rotations[r].x, -rotations[r].y),
                &buffer_a,
            );
        }

        rt.log_fps();
    });
}

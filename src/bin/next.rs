use std::marker::PhantomData;
use std::mem::swap;

use amitabha::color::BinaryF32;
use amitabha::storage::{BufferStorage, RadianceStorage};
use amitabha::trace::{AnalyticTracer, Circle, WorldMapper};
use amitabha::{DispatchAxis, Grid, MergeKernelSettings, Probe};
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::App;

fn main() {
    let grid_size = [2048, 2048];
    let app = App::new("Amitabha", grid_size)
        .scale(1)
        .dpi(2.0)
        .agx()
        .init();

    let mut buffer_a = DEVICE.create_buffer::<f32>((grid_size[0] * grid_size[1]) as usize);
    let mut buffer_b = DEVICE.create_buffer::<f32>((grid_size[0] * grid_size[1]) as usize);

    let tracer = WorldMapper {
        tracer: AnalyticTracer::<BinaryF32> {
            circles: vec![Circle {
                center: Vec2::splat(1024.0),
                radius: 16.0,
                color: 100.0,
            }],
        },
        world_origin: Vec2::splat(0.0),
        world_size: Vec2::splat(2048.0),
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

    let kernel = settings.build_kernel();

    let draw = DEVICE.create_kernel::<fn(Buffer<f32>)>(&track!(|buffer| {
        let cell = dispatch_id().xy();
        let radiance = BufferStorage.load(
            &buffer,
            Grid::new(Vec2::from(grid_size), 1).expr(),
            Probe::expr(cell, 0_u32.expr()),
        );
        app.display()
            .write(dispatch_id().xy(), Vec3::splat_expr(radiance));
    }));

    let num_cascades = 11;

    app.run(|rt, _scope| {
        for i in (0..num_cascades).rev() {
            swap(&mut buffer_a, &mut buffer_b);
            let grid = Grid::new(Vec2::new(grid_size[0], grid_size[1] >> i), 1 << i);
            kernel.dispatch(grid, &(), &buffer_a, &buffer_b).execute();
        }
        draw.dispatch([grid_size[0], grid_size[1], 1], &buffer_a);

        rt.log_fps();
    });
}

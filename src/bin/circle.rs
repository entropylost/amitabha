#![feature(more_float_constants)]

use amitabha::render::{HRCRenderer, HRCSettings};
use amitabha::trace::{AnalyticTracer, Circle};
use keter::graph::profile::Profiler;
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::App;

const DISPLAY_SIZE: u32 = 1024;
fn main() {
    let app = App::new("Amitabha", [DISPLAY_SIZE; 2])
        .scale(2048 / DISPLAY_SIZE)
        .agx()
        .init();
    let world = AnalyticTracer::new(&[
        Circle {
            center: Vec2::new(200.0, 512.0),
            radius: 10.0,
            color: Vec3::splat(f16::from_f32(1.0)),
        },
        Circle {
            center: Vec2::new(512.0, 512.0),
            radius: 10.0,
            color: Vec3::splat(f16::from_f32(0.0)),
        },
    ]);

    let renderer = HRCRenderer::new(
        &world,
        DISPLAY_SIZE,
        HRCSettings {
            traced_levels: u32::MAX,
        },
    );

    let draw = DEVICE.create_kernel::<fn()>(&track!(|| {
        set_block_size([8, 8, 1]);
        let cell = dispatch_id().xy();
        let radiance = Vec3::splat(f16::ZERO).var();
        for i in 0_u32..4_u32 {
            *radiance += renderer.radiance.read(cell.extend(i));
        }

        app.display().write(cell, radiance.cast_f32());
    }));

    let mut profiler = Profiler::new();

    app.run(|rt| {
        world.buffer.copy_from(&[
            Circle {
                center: rt.cursor_position,
                radius: 10.0,
                color: Vec3::splat(f16::from_f32(10.0)),
            },
            Circle {
                center: Vec2::new(512.0, 512.0),
                radius: 10.0,
                color: Vec3::splat(f16::from_f32(0.0)),
            },
        ]);
        profiler.record(
            (
                renderer.render(),
                draw.dispatch_async([DISPLAY_SIZE, DISPLAY_SIZE, 1])
                    .debug("Draw"),
            )
                .chain()
                .execute_timed(),
        );

        if profiler.time() > 3000.0 {
            profiler.print(&["Merge Up", "Merge Down", "Finish", "Draw"], true);
            profiler.reset();
        }
    });
}

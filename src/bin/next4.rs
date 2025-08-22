#![feature(more_float_constants)]

use std::f32::consts::PI;

use amitabha::color::{Color, Emission};
use amitabha::render::{HRCRenderer, HRCSettings};
use amitabha::trace::VoxelTracer;
use amitabha::{color, fluence};
use keter::graph::profile::Profiler;
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::{App, MouseButton};

const DISPLAY_SIZE: u32 = 512;

type F = fluence::RgbF16;
type C = color::RgbF16;

fn main() {
    let app = App::new("Amitabha", [DISPLAY_SIZE; 2])
        .scale(2048 / DISPLAY_SIZE)
        .agx()
        .init();
    let world = VoxelTracer::<C>::new(Vec2::new(DISPLAY_SIZE, DISPLAY_SIZE));

    let renderer = HRCRenderer::new(&world, DISPLAY_SIZE, HRCSettings::default());

    let draw = DEVICE.create_kernel::<fn()>(&track!(|| {
        set_block_size([8, 8, 1]);
        let cell = dispatch_id().xy();
        let base_fluence = world
            .read(cell)
            .to_fluence::<F>(0.5.expr())
            .restrict_angle((PI / 2.0).expr());
        let radiance = Vec3::splat(f16::ZERO).var();
        for i in 0_u32..4_u32 {
            *radiance += renderer.radiance.read(cell.extend(i));
        }
        let radiance = base_fluence.over_radiance(**radiance);

        app.display().write(cell, radiance.cast_f32());
    }));

    /*
    let julia = DEVICE.create_kernel::<fn()>(&track!(|| {
        let c = Vec2::<f32>::new(-0.835, -0.2321);
        let r = 2.0;
        assert!(r * r - r >= (c.x * c.x + c.y * c.y).sqrt());

        let pos = dispatch_id().xy().cast_f32() + 0.5;
        let pos = 2.0 * ((pos / dispatch_size().xy().cast_f32()) - Vec2::expr(0.5, 0.5)) * r * 0.7;
        let theta = 0.0_f32;
        let pos = Vec2::expr(
            pos.x * theta.cos() + pos.y * theta.sin(),
            -pos.x * theta.sin() + pos.y * theta.cos(),
        );
        let z = pos.var();

        let iter = u32::MAX.var();
        for i in 0_u32..1000 {
            *z = Vec2::expr(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
            if z.length() > r {
                *iter = i;
                break;
            }
        }
        let color = if iter > 30 {
            let iter = keter::min(iter, 1000);
            let j = iter.cast_f32() / 60.0;
            Color::<color::RgbF16>::expr(
                Vec3::<f32>::expr(0.1 * j, 0.0, 0.0).cast_f16(),
                Vec3::<f32>::splat_expr(0.3).cast_f16(),
            )
        } else {
            let j = iter.cast_f32() / 30.0;
            let l = if iter % 2 == 0 {
                1.0_f32.expr()
            } else {
                0.0.expr()
            };
            Color::expr(
                Vec3::<f32>::expr(0.0, 0.0, 0.0).cast_f16(),
                (Vec3::<f32>::expr(0.25, 1.0, 2.5) * l * j * j).cast_f16(),
            )
        };
        world.write(dispatch_id().xy(), color);
    }));

    julia.dispatch_blocking([DISPLAY_SIZE, DISPLAY_SIZE, 1]);
    */

    let rect_brush = DEVICE.create_kernel::<fn(Vec2<f32>, Vec2<f32>, Color<C>)>(&track!(
        |center, size, color| {
            let pos = dispatch_id().xy();
            if ((pos.cast_f32() + 0.5 - center).abs() < size).all() {
                world.write(pos, color);
            }
        }
    ));
    let circle_brush =
        DEVICE.create_kernel::<fn(Vec2<f32>, f32, Color<C>)>(&track!(|center, radius, color| {
            let pos = dispatch_id().xy();
            if (pos.cast_f32() + 0.5 - center).length() < radius {
                world.write(pos, color);
            }
        }));

    circle_brush.dispatch(
        [DISPLAY_SIZE, DISPLAY_SIZE, 1],
        &Vec2::new(256.0, 256.0),
        &10.0, // &Vec2::new(10.0, 10.0),
        &Color::new(
            Vec3::splat(f16::from_f32(5.0)),
            Vec3::splat(f16::from_f32(0.5)),
        ),
    );

    let mut profiler = Profiler::new();

    app.run(|rt| {
        let brushes = [
            (
                MouseButton::Middle,
                Color::new(
                    Vec3::splat(f16::from_f32(1.0)),
                    Vec3::splat(f16::from_f32(0.5)),
                ),
            ),
            (MouseButton::Left, Color::solid(Vec3::black())),
            (MouseButton::Right, Color::empty()),
            (
                MouseButton::Back,
                Color::new(
                    Vec3::new(f16::ZERO, f16::ONE, f16::ONE),
                    Vec3::splat(f16::from_f32(0.5)),
                ),
            ),
        ];
        for brush in brushes {
            if rt.button_down(brush.0) {
                circle_brush.dispatch(
                    [world.size.x, world.size.y, 1],
                    &rt.cursor_position,
                    &6.0,
                    &brush.1,
                );
            }
        }

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

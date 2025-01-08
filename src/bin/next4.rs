#![feature(more_float_constants)]

use std::f32::consts::PI;
use std::time::Instant;

use amitabha::color::{Color, Emission};
use amitabha::render::{HRCRenderer, HRCSettings};
use amitabha::trace::VoxelTracer;
use amitabha::{color, fluence};
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::{App, MouseButton};

const DISPLAY_SIZE: u32 = 1024;

type F = fluence::RgbF16;
type C = color::RgbF16;

fn main() {
    let grid_size = [DISPLAY_SIZE; 2];
    let app = App::new("Amitabha", grid_size)
        .scale(2048 / DISPLAY_SIZE)
        .dpi(2.0)
        .agx()
        .init();
    let world = VoxelTracer::<C>::new(Vec2::new(DISPLAY_SIZE, DISPLAY_SIZE));

    let renderer = HRCRenderer::new(&world, DISPLAY_SIZE, HRCSettings { traced_levels: 2 });

    let draw = DEVICE.create_kernel::<fn()>(&track!(|| {
        set_block_size([8, 8, 1]);
        let cell = dispatch_id().xy();
        let base_fluence = world
            .read(cell)
            .to_fluence::<F>(0.5.expr())
            .restrict_angle((PI / 2.0).expr());
        let radiance = base_fluence.over_radiance(renderer.radiance.read(cell));

        app.display().write(cell, radiance.cast_f32());
    }));

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

    rect_brush.dispatch(
        [DISPLAY_SIZE, DISPLAY_SIZE, 1],
        &Vec2::new(0.0, 0.0),
        &Vec2::new(10000.0, 10.0),
        &Color::new(
            Vec3::splat(f16::from_f32(5.0)),
            Vec3::splat(f16::from_f32(0.5)),
        ),
    );

    app.run(|rt, _scope| {
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
            if rt.pressed_button(brush.0) {
                circle_brush.dispatch(
                    [world.size.x, world.size.y, 1],
                    &rt.cursor_position,
                    &6.0,
                    &brush.1,
                );
            }
        }

        let start = Instant::now();
        renderer.render().execute_blocking();
        println!("Render: {:?}", start.elapsed());
        draw.dispatch([DISPLAY_SIZE, DISPLAY_SIZE, 1]);
    });
}

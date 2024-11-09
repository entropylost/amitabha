use std::f32::consts::TAU;

use glam::IVec2;
use glam::UVec2;
use glam::Vec2 as FVec2;
use glam::Vec2Swizzles;
use image::ImageReader;
use luisa::lang::types::vector::Vec4;
use luisa::lang::types::vector::{Vec2, Vec3};
use sefirot::prelude::*;
use sefirot_testbed::App;
use sefirot_testbed::KeyCode;
use sefirot_testbed::MouseButton;

fn main() {
    let grid_size = [2048, 2048];
    let app = App::new("Amitabha", grid_size)
        .scale(1)
        .dpi(2.0)
        .agx()
        .init();

    let dim = 128;

    let factor = (grid_size[0] / dim) as f32;
    let corner = -(FVec2::new(1024.0 - 64.0 + 16.0, 1123.0) - FVec2::splat(dim as f32 / 2.0))
        * factor
        + FVec2::splat(factor / 2.0);

    let bg = ImageReader::open("diff256.png")
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb32f();
    let mut bg2 = vec![];
    for j in 64..192 {
        for i in 16..128 + 16 {
            bg2.push(Vec3::from(bg.get_pixel(i, j).0));
        }
    }
    let bg = DEVICE.create_buffer_from_slice(&bg2);

    let lines_endpoints = [487, 268, 49, 1024, 1565, 1784, 2000];

    let draw_bg = DEVICE.create_kernel::<fn()>(&track!(|| {
        let pos = dispatch_id().xy();
        app.display().write(
            pos,
            bg.read(pos.x / (factor as u32) + pos.y / (factor as u32) * dim),
        );
    }));

    let draw_line_t = DEVICE.create_kernel::<fn(Vec2<f32>, Vec2<f32>, Vec3<f32>)>(&track!(
        |start, end, color| {
            let t = dispatch_id().x.cast_f32() / dispatch_size().x.cast_f32();
            let pos = start * (1.0 - t) + end * t;
            app.display().write(pos.cast_u32(), color);
        }
    ));
    let draw_line = |start: FVec2, end: FVec2, color: Vec3<f32>, spacing: f32| {
        let start = start * factor + corner;
        let end = end * factor + corner;
        draw_line_t.dispatch(
            [((start - end).length() * spacing).ceil() as u32, 1, 1],
            &Vec2::from(start),
            &Vec2::from(end),
            &color,
        );
    };

    let draw_circle_t =
        DEVICE.create_kernel::<fn(Vec2<f32>, f32, Vec3<f32>)>(&track!(|center, radius, color| {
            let pos = dispatch_id().xy();
            if (pos.cast_f32() - center).length() < radius {
                app.display().write(pos, color);
            }
        }));
    let draw_circle = |center: FVec2, radius: f32, color: Vec3<f32>| {
        draw_circle_t.dispatch(
            [grid_size[0], grid_size[1], 1],
            &Vec2::from(center * factor + corner),
            &(radius * factor),
            &color,
        );
    };

    let colors = [
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];

    let mut display_frustrums = false;

    let mut cell = IVec2::new(30, 0);

    let light = (FVec2::new(1024.0, 1123.0), 4.0);

    app.run(|rt, _scope| {
        draw_bg.dispatch([grid_size[0] as u32, grid_size[1] as u32, 1]);

        if rt.just_pressed_key(KeyCode::KeyF) {
            display_frustrums = !display_frustrums;
        }

        if rt.pressed_button(MouseButton::Left) {
            cell = ((FVec2::from(rt.cursor_position).yx() - corner.yx()) / factor)
                .round()
                .as_ivec2();
        }

        let mut all_rays = vec![
            (
                cell.yx().as_vec2(),
                cell.yx().as_vec2() + FVec2::new(1000.0, -1000.0),
                Vec3::splat(5.0),
                1.0,
            ),
            (
                cell.yx().as_vec2(),
                cell.yx().as_vec2() + FVec2::new(1000.0, 1000.0),
                Vec3::splat(5.0),
                1.0,
            ),
        ];
        if display_frustrums {
            all_rays.clear();
        }
        let cell = cell + IVec2::Y;

        let mut rays = vec![(0_i32, cell)];

        let axis_x = FVec2::new(0.0, 1.0);

        for c in 0..7 {
            let mut next_rays = vec![];
            for (angle, cell) in rays {
                let axis_y = FVec2::new((1 << c) as f32, 0.0);
                let angle_res = 1_i32 << c;

                all_rays.push((
                    axis_y * cell.y as f32 + axis_x * cell.x as f32,
                    axis_y * (cell.y as f32 + 1.0)
                        + axis_x * (cell.x + 2 * angle - angle_res + 1) as f32,
                    colors[c],
                    1.0,
                ));

                let f = if display_frustrums { 1 } else { 0 };

                // all_rays.push((
                //     axis_y * cell.y as f32 + axis_x * (cell.x as f32 - f as f32 * 1.0),
                //     axis_y * (cell.y as f32 + 1.0)
                //         + axis_x * (cell.x + 2 * angle - angle_res - f) as f32,
                //     colors[c],
                //     0.1,
                // ));
                // all_rays.push((
                //     axis_y * cell.y as f32 + axis_x * (cell.x as f32 + f as f32 * 1.0),
                //     axis_y * (cell.y as f32 + 1.0)
                //         + axis_x * (cell.x + 2 * angle - angle_res + 2 + f) as f32,
                //     colors[c],
                //     0.1,
                // ));

                let offset_0 = 2 * angle - angle_res;
                let offset_1 = 2 * angle - angle_res + 2;

                if cell.y.rem_euclid(2) == 0 {
                    next_rays.push((angle * 2, IVec2::new(cell.x, cell.y / 2)));
                    next_rays.push((angle * 2 + 1, IVec2::new(cell.x, cell.y / 2)));
                    next_rays.push((angle * 2, IVec2::new(cell.x + offset_0 * 2, cell.y / 2 + 1)));
                    next_rays.push((
                        angle * 2 + 1,
                        IVec2::new(cell.x + offset_1 * 2, cell.y / 2 + 1),
                    ));
                } else {
                    next_rays.push((
                        angle * 2,
                        IVec2::new(cell.x + offset_0, cell.y.div_euclid(2) + 1),
                    ));
                    next_rays.push((
                        angle * 2 + 1,
                        IVec2::new(cell.x + offset_1, cell.y.div_euclid(2) + 1),
                    ));
                }
            }
            rays = next_rays;
        }

        // for l in &lines_endpoints {
        //     let wpos_l = *l as f32 + (1123.0 - 1024.0);
        //     draw_line(light.0, FVec2::new(0.0, wpos_l), Vec3::splat(5.0), 1.0);
        // }

        for ray in &all_rays {
            draw_line(ray.0, ray.1, ray.2, ray.3);
        }
        draw_circle(light.0, light.1, Vec3::splat(5.0));
    });
}

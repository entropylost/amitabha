use std::f32::consts::TAU;

use luisa::lang::types::vector::{Vec2, Vec3};
use sefirot::prelude::*;
use sefirot_testbed::App;
use utils::pcgf;

mod utils;

#[tracked]
fn intersect_circle(
    ray_start: Expr<Vec2<f32>>,
    ray_dir: Expr<Vec2<f32>>,
    radius: Expr<f32>,
) -> (Expr<f32>, Expr<f32>, Expr<bool>) {
    let dist_to_parallel = -ray_start.dot(ray_dir);
    let min_point = ray_start + dist_to_parallel * ray_dir;
    let dist_to_center = min_point.length();
    if dist_to_center > radius {
        (0.0.expr(), 0.0.expr(), false.expr())
    } else {
        let dist_to_intersection = (radius.sqr() - dist_to_center.sqr()).sqrt();
        let min_t = dist_to_parallel - dist_to_intersection;
        let max_t = dist_to_parallel + dist_to_intersection;
        (min_t, max_t, true.expr())
    }
}

// color, t
#[tracked]
fn trace(
    ray_dir: Expr<Vec2<f32>>,
    ray_start: Expr<Vec2<f32>>,
    interval: Expr<Vec2<f32>>,
) -> (Expr<f32>, Expr<f32>) {
    let circles = [
        (Vec2::new(10.0, 10.0), 4.0, 100.0),
        (Vec2::new(200.0, 200.0), 10.0, 0.0),
    ];
    let best_t = interval.y.var();
    let best_color = 0.0_f32.var();
    for (center, radius, color) in circles.iter() {
        let (min_t, max_t, hit) =
            intersect_circle(ray_start - center.expr(), ray_dir, radius.expr());
        if hit && max_t > interval.x && min_t < best_t {
            *best_t = min_t;
            *best_color = *color;
        }
    }
    (**best_color, **best_t)
}

fn main() {
    let grid_size = [512, 512];
    let app = App::new("Amitabha", grid_size)
        .scale(4)
        .dpi(2.0)
        .agx()
        .init();

    let trace_kernel = DEVICE.create_kernel::<fn()>(&track!(|| {
        let num_dirs = grid_size[0] * 6;

        let offset = pcgf(dispatch_id().x + dispatch_id().y * grid_size[0]);

        let total_color = 0.0_f32.var();
        let pos = dispatch_id().xy().cast_f32();
        for i in 0_u32.expr()..num_dirs.expr() {
            let i: Expr<u32> = i;
            let ray_angle = (i.cast_f32() + offset) / num_dirs as f32 * TAU;
            let ray_dir = Vec2::expr(ray_angle.cos(), ray_angle.sin());
            let (color, _t) = trace(ray_dir, pos, Vec2::new(0.0, 1000.0).expr());
            *total_color += color;
        }
        app.display().write(
            dispatch_id().xy(),
            Vec3::splat_expr(total_color / num_dirs as f32),
        );
    }));

    app.run(|rt, scope| {
        trace_kernel.dispatch([grid_size[0], grid_size[1], 1]);
    });
}

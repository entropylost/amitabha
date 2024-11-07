use std::f32::consts::TAU;

use glam::IVec2;
use glam::UVec2;
use glam::Vec2 as FVec2;
use luisa::lang::types::vector::{Vec2, Vec3};
use sefirot::prelude::*;
use sefirot_testbed::App;
use sefirot_testbed::KeyCode;
use sefirot_testbed::MouseButton;
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
) -> (Expr<f32>, Expr<bool>) {
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
    (**best_color, best_t < interval.y)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
struct RayData {
    color: f32,
}

struct GridStorage {
    grids: Buffer<Grid2>,
    base_grids: u32,
    data: Buffer<RayData>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
struct Grid2 {
    origin: Vec2<f32>,
    axis_x: Vec2<f32>,
    axis_y: Vec2<f32>,
    size: Vec2<u32>, // From -x/2=..x/2.
    ray_angle: f32,
    angle_resolution: u32,
    ray_interval: Vec2<f32>,
    data_offset: u32,
}

#[derive(Debug, Clone, Copy)]
struct Grid {
    origin: FVec2,
    axis_x: FVec2,
    axis_y: FVec2,
    size: UVec2, // From -x..x.
    ray_angle: f32,
    angle_resolution: u32,
    ray_interval: FVec2,
}

impl Grid {
    fn first_level(ray_angle: f32) -> Self {
        let c = 4.0;
        let ray_dir = FVec2::from_angle(ray_angle) * c;
        Self {
            origin: FVec2::new(1024.0, 1024.0),
            axis_x: FVec2::new(ray_dir.y, -ray_dir.x),
            axis_y: ray_dir,
            size: UVec2::new(256, 256),
            ray_angle,
            angle_resolution: 4,
            ray_interval: FVec2::new(4.0, 8.0) * c,
        }
    }
    fn ray_dir(&self) -> FVec2 {
        FVec2::from_angle(self.ray_angle)
    }
    fn to_world(&self, pos: FVec2) -> FVec2 {
        self.origin + self.axis_x * pos.x + self.axis_y * pos.y
    }
    fn from_world(&self, pos: FVec2) -> FVec2 {
        let local = pos - self.origin;
        FVec2::new(
            local.dot(self.axis_x.normalize()) / self.axis_x.length(),
            local.dot(self.axis_y.normalize()) / self.axis_y.length(),
        )
    }
    fn split_level(&self, left: bool) -> Self {
        let next_angle_resolution = self.angle_resolution * 2;
        let next_angle = self.ray_angle
            + TAU / 2.0 / next_angle_resolution as f32 * if left { -1.0 } else { 1.0 };
        let next_axis_x_dir = FVec2::from_angle(next_angle + TAU / 4.0);
        let next_axis_x =
            self.axis_x.length_squared() / next_axis_x_dir.dot(self.axis_x) * next_axis_x_dir;
        let next_axis_y = (self.axis_y
            * (next_axis_x.length() / self.axis_y.dot(next_axis_x_dir).abs()))
        .project_onto_normalized(FVec2::from_angle(next_angle));
        let next_ray_interval = self.ray_interval * 2.0;
        let next_size = UVec2::new(self.size.x, self.size.y / 2);
        let next_origin = self.origin;
        Self {
            origin: next_origin,
            axis_x: next_axis_x,
            axis_y: next_axis_y,
            size: next_size,
            ray_angle: next_angle,
            angle_resolution: next_angle_resolution,
            ray_interval: next_ray_interval,
        }
    }
}

fn main() {
    let num_cascades = 6;

    let grid_size = [2048, 2048];
    let app = App::new("Amitabha", grid_size)
        .scale(1)
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

    let mut grids_host_lv = (0..4)
        .map(|i| Grid::first_level(i as f32 * TAU / 4.0))
        .collect::<Vec<_>>();
    let mut grids_host = vec![];
    for _c in 0..num_cascades {
        let mut next_grids = vec![];
        for grid in grids_host_lv {
            grids_host.push(grid);
            for l in [false, true] {
                next_grids.push(grid.split_level(l));
            }
        }
        grids_host_lv = next_grids;
    }

    let mut data_size = 0;
    let grids_host = grids_host
        .iter()
        .map(|g| {
            let grid = Grid2 {
                origin: g.origin.into(),
                axis_x: g.axis_x.into(),
                axis_y: g.axis_y.into(),
                size: g.size.into(),
                ray_angle: g.ray_angle,
                angle_resolution: g.angle_resolution,
                ray_interval: g.ray_interval.into(),
                data_offset: data_size,
            };
            data_size += g.size.x * g.size.y;
            grid
        })
        .collect::<Vec<_>>();

    let data = DEVICE.create_buffer_from_fn(data_size as usize, |i| RayData { color: 0.0 });
    let grids = DEVICE.create_buffer_from_slice(&grids_host);

    let storage = GridStorage {
        grids,
        base_grids: 4,
        data,
    };

    // let merge = DEVICE.create_kernel::<fn(u32)>(&track!(|lv| {
    //     let num_grids = 4 << lv;
    //     let grid = storage.grid_at(lv, )
    // }));

    let draw_grid = DEVICE.create_kernel::<fn(Vec2<f32>, Vec2<f32>, Vec2<f32>, Vec3<f32>)>(
        &track!(|origin, axis_x, axis_y, color| {
            let cell = dispatch_id().xy().cast_i32() - dispatch_size().xy().cast_i32() / 2;
            let pos = origin + axis_x * cell.x.cast_f32() + axis_y * cell.y.cast_f32();
            app.display().write(pos.cast_u32(), color);
        }),
    );
    let draw_line_t = DEVICE.create_kernel::<fn(Vec2<f32>, Vec2<f32>, Vec3<f32>)>(&track!(
        |start, end, color| {
            let t = dispatch_id().x.cast_f32() / dispatch_size().x.cast_f32();
            let pos = start * (1.0 - t) + end * t;
            app.display().write(pos.cast_u32(), color);
        }
    ));
    let draw_line = |start: FVec2, end: FVec2, color: Vec3<f32>| {
        draw_line_t.dispatch([100, 1, 1], &Vec2::from(start), &Vec2::from(end), &color);
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

    // println!("All rays: {:#?}", all_rays);

    let mut start_pos = FVec2::new(1024.0, 1024.0);

    app.run(|rt, scope| {
        // if rt.just_pressed_key(KeyCode::Equal) {
        //     println!("Last: {:?}", grid);
        //     grid = grid.split_level(false);
        // }

        if rt.pressed_button(MouseButton::Left) {
            start_pos = rt.cursor_position.into();
        }

        let mut all_rays = vec![];

        let mut rays = (0..4)
            .map(|i| {
                let grid = Grid::first_level(i as f32 * TAU / 4.0);
                (grid, grid.from_world(start_pos).round().as_ivec2())
            })
            .collect::<Vec<_>>();

        // println!("Start pos: {:?}, {:?}", start_pos, rays[0].1);

        for c in 0..6 {
            let mut next_rays = vec![];
            for &(grid, coords) in &rays {
                let pos = grid.to_world(coords.as_vec2());
                let start = pos + grid.ray_dir() * grid.ray_interval.x;
                let end = pos + grid.ray_dir() * grid.ray_interval.y;
                if coords.abs().cmplt(grid.size.as_ivec2() / 2).all() {
                    all_rays.push((start, end, colors[c]));
                }
                for l in [false, true] {
                    let next_grid = grid.split_level(l);
                    let next_pos = pos;
                    let next_coords = next_grid.from_world(next_pos);

                    let (nc_a, nc_b) = if !l {
                        (
                            FVec2::new(next_coords.x.floor(), next_coords.y.floor()),
                            FVec2::new(next_coords.x.floor() + 1.0, next_coords.y.floor() + 1.0),
                        )
                    } else {
                        (
                            FVec2::new(next_coords.x.floor(), next_coords.y.floor() + 1.0),
                            FVec2::new(next_coords.x.floor() + 1.0, next_coords.y.floor()),
                        )
                    };

                    next_rays.push((next_grid, nc_a.as_ivec2()));
                    next_rays.push((next_grid, nc_b.as_ivec2()));
                }
            }
            rays = next_rays;
        }

        for ray in &all_rays {
            draw_line(ray.0, ray.1, ray.2);
        }

        /*

        draw_grid.dispatch(
            [grid.size.x, grid.size.y, 1],
            &Vec2::from(grid.origin),
            &Vec2::from(grid.axis_x),
            &Vec2::from(grid.axis_y),
            &colors[0],
        );

        let grid = grid.split_level(false);

        draw_grid.dispatch(
            [grid.size.x, grid.size.y, 1],
            &Vec2::from(grid.origin),
            &Vec2::from(grid.axis_x),
            &Vec2::from(grid.axis_y),
            &colors[1],
        );

        let grid = grid.split_level(false);

        draw_grid.dispatch(
            [grid.size.x, grid.size.y, 1],
            &Vec2::from(grid.origin),
            &Vec2::from(grid.axis_x),
            &Vec2::from(grid.axis_y),
            &colors[2],
        );

        let grid = grid.split_level(false);

        draw_grid.dispatch(
            [grid.size.x, grid.size.y, 1],
            &Vec2::from(grid.origin),
            &Vec2::from(grid.axis_x),
            &Vec2::from(grid.axis_y),
            &colors[3],
        );

         */

        //  draw_line(
        //      grid.origin + grid.ray_dir() * grid.ray_interval.x,
        //      grid.origin + grid.ray_dir() * grid.ray_interval.y,
        //      Vec3::new(1.0, 0.0, 0.0),
        //  );
        // trace_kernel.dispatch([grid_size[0], grid_size[1], 1]);
    });
}

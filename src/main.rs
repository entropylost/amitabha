#![allow(clippy::wrong_self_convention)]
#![allow(clippy::needless_range_loop)]

use std::f32::consts::TAU;

use glam::{IVec2, UVec2, Vec2 as FVec2, Vec2Swizzles};
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::{App, KeyCode, MouseButton};
use utils::pcgf;

mod utils;

type TraceArgs = (Expr<Vec2<f32>>, Expr<f32>);

#[tracked]
fn intersect_circle(
    ray_start: Expr<Vec2<f32>>,
    ray_dir: Expr<Vec2<f32>>,
    radius: Expr<f32>,
) -> (Expr<f32>, Expr<f32>, Expr<f32>, Expr<bool>) {
    let dist_to_parallel = -ray_start.dot(ray_dir);
    let min_point = ray_start + dist_to_parallel * ray_dir;
    let dist_to_center = min_point.length();
    let penetration = radius - dist_to_center;
    if penetration < 0.0.expr() {
        (0.0.expr(), 0.0.expr(), penetration, false.expr())
    } else {
        let dist_to_intersection = (radius.sqr() - dist_to_center.sqr()).sqrt();
        let min_t = dist_to_parallel - dist_to_intersection;
        let max_t = dist_to_parallel + dist_to_intersection;
        (min_t, max_t, penetration, true.expr())
    }
}

// color, t
#[tracked]
fn trace(
    ray_dir: Expr<Vec2<f32>>,
    ray_start: Expr<Vec2<f32>>,
    interval: Expr<Vec2<f32>>,
    args: TraceArgs,
) -> (Expr<f32>, Expr<f32>) {
    let circles = [
        (args.0, args.1, 100.0),
        (Vec2::expr(1000.0, 1200.0), 16.0.expr(), 0.0),
    ];
    let best_t = interval.y.var();
    let best_color = 0.0_f32.var();
    for (center, radius, color) in circles.iter() {
        let (min_t, max_t, penetration, hit) =
            intersect_circle(ray_start - center, ray_dir, *radius);
        if hit && max_t > interval.x && min_t < best_t {
            *best_t = min_t;
            *best_color = *color * (keter::min(penetration / 2.0, 1.0));
        }
    }
    (
        **best_color,
        if best_t < interval.y {
            0.0_f32.expr()
        } else {
            1.0_f32.expr()
        },
    )
}

#[tracked]
fn over2(
    (color, transmittance): (Expr<f32>, Expr<f32>),
    (next_color, next_transmittance): (Expr<f32>, Expr<f32>),
) -> (Expr<f32>, Expr<f32>) {
    (
        color + next_color * transmittance,
        transmittance * next_transmittance,
    )
}

#[tracked]
fn over((color, transmittance): (Expr<f32>, Expr<f32>), next_color: Expr<f32>) -> Expr<f32> {
    color + next_color * transmittance
}

#[tracked]
fn trace_between(
    start: Expr<Vec2<f32>>,
    end: Expr<Vec2<f32>>,
    args: TraceArgs,
) -> (Expr<f32>, Expr<f32>) {
    let delta = end - start;
    let l = delta.length();
    let ray_dir = delta / l;
    let interval = Vec2::expr(0.0, l);
    trace(ray_dir, start, interval, args)
}

#[tracked]
fn trace_frustrum_b(
    start_a: Expr<Vec2<f32>>,
    end_a: Expr<Vec2<f32>>,
    start_b: Expr<Vec2<f32>>,
    end_b: Expr<Vec2<f32>>,
    args: TraceArgs,
) -> (Expr<f32>, Expr<f32>) {
    let color = 0.0_f32.var();
    let transmittance = 0.0_f32.var();
    let (c, t) = trace_between(start_a, end_a, args);
    *color += c;
    *transmittance += t;
    let (c, t) = trace_between(start_b, end_b, args);
    *color += c;
    *transmittance += t;
    let (c, t) = trace_between(start_a, end_b, args);
    *color += c;
    *transmittance += t;
    let (c, t) = trace_between(start_b, end_a, args);
    *color += c;
    *transmittance += t;
    (color / 4.0, transmittance / 4.0)
}

#[tracked]
fn trace_frustrum(
    start_a: Expr<Vec2<f32>>,
    end_a: Expr<Vec2<f32>>,
    start_b: Expr<Vec2<f32>>,
    end_b: Expr<Vec2<f32>>,
    args: TraceArgs,
) -> (Expr<f32>, Expr<f32>) {
    let num_samples = 1_u32;
    let color = 0.0_f32.var();
    let transmittance = 0.0_f32.var();
    for i in 0_u32..num_samples {
        let i: Expr<u32> = i;
        let t = (i.cast_f32() + 0.5) / (num_samples as f32);
        let start = start_a * t + start_b * (1.0 - t);
        let end = end_a * t + end_b * (1.0 - t);

        let (c, t) = trace_between(start, end, args);
        *color += c;
        *transmittance += t;
    }
    (
        color / num_samples as f32,
        transmittance / num_samples as f32,
    )
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
struct MergeUpRayData {
    trace_transmittance: f32,
    trace_color: f32,
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
    merge_up_data: Buffer<MergeUpRayData>,
}
impl GridStorage {
    #[tracked]
    fn grid_at(&self, lv: Expr<u32>, dir: Expr<u32>) -> Expr<Grid2> {
        let offset = lv * self.base_grids + dir;
        self.grids.read(offset)
    }

    #[tracked]
    fn load_grid_up(
        &self,
        grid: Expr<Grid2>,
        pos: Expr<Vec2<i32>>,
        angle: Expr<u32>,
    ) -> (Expr<f32>, Expr<f32>) {
        let cell = grid.cell_index(pos, angle);
        if cell != u32::MAX {
            let data = self.merge_up_data.read(cell);
            (data.trace_color, data.trace_transmittance)
        } else {
            (0.0_f32.expr(), 1.0_f32.expr())
        }
    }

    #[tracked]
    fn load_grid(&self, grid: Expr<Grid2>, pos: Expr<Vec2<i32>>, angle: Expr<u32>) -> Expr<f32> {
        let cell = grid.cell_index(pos, angle);
        if cell != u32::MAX {
            self.data.read(cell).color
        } else {
            0.0_f32.expr()
        }
    }

    #[tracked]
    fn load_grid_bilinear(
        &self,
        grid: Expr<Grid2>,
        x: Expr<f32>,
        y: Expr<i32>,
        angle: Expr<u32>,
    ) -> Expr<f32> {
        let a = (x.round() - x).abs() < 0.01;
        lc_assert!(a);
        let x_0 = x.floor().cast_i32();
        let x_1 = x_0 + 1;
        let a = x - x_0.cast_f32();
        let b = 1.0 - a;

        self.load_grid(grid, Vec2::expr(x_0, y), angle) * b
            + self.load_grid(grid, Vec2::expr(x_1, y), angle) * a
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
struct Grid2 {
    origin: Vec2<f32>,
    axis_x: Vec2<f32>,
    axis_y: Vec2<f32>,
    size: Vec2<u32>,
    angle_resolution: u32,
    data_offset: u32,
}
impl Grid2Expr {
    #[tracked]
    fn cell_index(&self, cell: Expr<Vec2<i32>>, angle: Expr<u32>) -> Expr<u32> {
        if (cell < 0).any() || (cell >= self.size.cast_i32()).any() {
            u32::MAX.expr()
        } else {
            let cell = cell.cast_u32();
            self.data_offset + cell.x + cell.y * self.size.x + angle * self.size.x * self.size.y
        }
    }

    #[tracked]
    fn ray_pos(&self, angle: Expr<f32>) -> Expr<f32> {
        2.0 * angle - self.angle_resolution.cast_f32() + 1.0
    }

    #[tracked]
    fn ray_angle(&self, angle: Expr<f32>) -> Expr<f32> {
        let y_pos = self.ray_pos(angle);
        let x_pos = self.axis_y.length() / self.axis_x.length();
        y_pos.atan2(x_pos)
    }

    #[tracked]
    fn angle_size(&self, angle: Expr<u32>) -> Expr<f32> {
        let upper = self.ray_angle(angle.cast_f32() + 0.5);
        let lower = self.ray_angle(angle.cast_f32() - 0.5);
        upper - lower
    }

    #[tracked]
    fn ray_dir(&self, angle: Expr<f32>) -> Expr<Vec2<f32>> {
        let angle = self.ray_angle(angle);
        let ray_dir = Vec2::expr(angle.cos(), angle.sin());
        let axis = self.axis_y.normalize();
        Vec2::expr(
            ray_dir.x * axis.x - ray_dir.y * axis.y,
            ray_dir.y * axis.x + ray_dir.x * axis.y,
        )
    }

    #[tracked]
    fn ray_len(&self, dir: Expr<Vec2<f32>>) -> Expr<f32> {
        self.axis_y.length_squared() / (dir.dot(self.axis_y))
    }

    #[tracked]
    fn to_world(&self, pos: Expr<Vec2<f32>>) -> Expr<Vec2<f32>> {
        self.origin + self.axis_x * pos.x + self.axis_y * pos.y
    }
    #[tracked]
    fn from_world(&self, pos: Expr<Vec2<f32>>) -> Expr<Vec2<f32>> {
        let local = pos - self.origin;
        Vec2::expr(
            local.dot(self.axis_x.normalize()) / self.axis_x.length(),
            local.dot(self.axis_y.normalize()) / self.axis_y.length(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct Grid {
    origin: FVec2,
    axis_x: FVec2,
    axis_y: FVec2,
    size: UVec2, // From -x..x.
    angle_resolution: u32,
}

const BASE_SIZE: UVec2 = UVec2::new(2048, 2048);

impl Grid {
    fn data_size(&self) -> u32 {
        self.size.x * self.size.y * self.angle_resolution
    }
    fn first_level(ray_dir: FVec2) -> Self {
        let axis_x = ray_dir.rotate(FVec2::new(0.0, 1.0));
        let axis_y = ray_dir.rotate(FVec2::new(1.0, 0.0));
        Self {
            origin: FVec2::new(1024.0, 1024.0)
                - axis_x * BASE_SIZE.x as f32 / 2.0
                - axis_y * BASE_SIZE.y as f32 / 2.0,
            axis_x,
            axis_y,
            size: BASE_SIZE,
            angle_resolution: 1,
        }
    }
    fn next_level(&self) -> Self {
        Self {
            origin: self.origin,
            axis_x: self.axis_x,
            axis_y: self.axis_y * 2.0,
            size: UVec2::new(self.size.x, self.size.y / 2),
            angle_resolution: self.angle_resolution * 2,
        }
    }
}

fn main() {
    let num_cascades = BASE_SIZE.max_element().trailing_zeros();
    println!("Num cascades: {}", num_cascades);

    let grid_size = [2048, 2048];
    let app = App::new("Amitabha", grid_size)
        .scale(1)
        .dpi(2.0)
        .agx()
        .init();

    let mut light_pos = Vec2::new(1024.0, 1123.0);
    let focus = Vec2::new(1024.0, 1024.0);
    let corner = Vec2::new(
        focus.x - grid_size[0] as f32 / 2.0,
        focus.y - grid_size[1] as f32 / 2.0,
    );

    let trace2_kernel = DEVICE.create_kernel::<fn(Vec2<f32>, f32)>(&track!(|vpos, r| {
        let total_color = 0.0_f32.var();
        let pos = dispatch_id().xy().cast_f32() + corner;
        for i in 0_u32.expr()..grid_size[0].expr() {
            let i: Expr<u32> = i;
            let ray_pos = (2.0 * i.cast_f32() + 1.0) / grid_size[0] as f32 - 1.0;

            let upper_ray_pos = (2.0 * i.cast_f32() + 2.0) / grid_size[0] as f32 - 1.0;
            let lower_ray_pos = 2.0 * i.cast_f32() / grid_size[0] as f32 - 1.0;

            let angle_size = upper_ray_pos.atan() - lower_ray_pos.atan();

            let ray_dir = Vec2::expr(1.0, ray_pos).normalize();
            let (color, _t) = trace(
                ray_dir,
                pos,
                Vec2::new(0.0, f32::INFINITY).expr(),
                (vpos, r),
            );
            *total_color += color * angle_size;
        }
        app.display()
            .write(dispatch_id().xy(), Vec3::splat_expr(total_color));
    }));

    let trace_kernel = DEVICE.create_kernel::<fn(Vec2<f32>, f32)>(&track!(|vpos, r| {
        let num_dirs = grid_size[0] * 6;

        let offset = pcgf(dispatch_id().x + dispatch_id().y * grid_size[0]);

        let total_color = 0.0_f32.var();
        let pos = dispatch_id().xy().cast_f32() + corner;
        for i in 0_u32.expr()..num_dirs.expr() {
            let i: Expr<u32> = i;
            let ray_angle = (i.cast_f32() + offset) / num_dirs as f32 * TAU;
            let ray_dir = Vec2::expr(ray_angle.cos(), ray_angle.sin());
            let (color, _t) = trace(
                ray_dir,
                pos,
                Vec2::new(0.0, f32::INFINITY).expr(),
                (vpos, r),
            );
            *total_color += color;
        }
        app.display().write(
            dispatch_id().xy(),
            Vec3::splat_expr(total_color / num_dirs as f32),
        );
    }));

    let mut grids_host_lv = [FVec2::X, FVec2::Y, FVec2::NEG_X, FVec2::NEG_Y]
        .into_iter()
        .map(|i| Grid::first_level(i))
        .collect::<Vec<_>>();
    let mut grids_host = vec![];
    for _c in 0..=num_cascades {
        let mut next_grids = vec![];
        for grid in grids_host_lv {
            grids_host.push(grid);
            next_grids.push(grid.next_level());
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
                angle_resolution: g.angle_resolution,
                data_offset: data_size,
            };
            data_size += g.data_size();
            grid
        })
        .collect::<Vec<_>>();

    let data = DEVICE.create_buffer_from_fn(data_size as usize, |_i| RayData { color: 0.0 });
    let merge_up_data = DEVICE.create_buffer_from_fn(data_size as usize, |_i| MergeUpRayData {
        trace_transmittance: 1.0,
        trace_color: 0.0,
    });
    let grids = DEVICE.create_buffer_from_slice(&grids_host);

    let storage = GridStorage {
        grids,
        base_grids: 4,
        data,
        merge_up_data,
    };

    let init_merge_up = DEVICE.create_kernel::<fn(u32, (Vec2<f32>, f32))>(&track!(|face, r| {
        let grid = storage.grid_at(0.expr(), face);
        let cell = dispatch_id().xy().cast_i32();

        let data_index = grid.cell_index(cell, 0.expr());
        let a = data_index != u32::MAX;
        lc_assert!(a);

        let start = grid.to_world(cell.cast_f32());
        let dir = grid.ray_dir(0_f32.expr());
        let tr = trace(dir, start, Vec2::expr(0.0, grid.ray_len(dir)), r);

        storage.merge_up_data.write(
            data_index,
            MergeUpRayData::from_comps_expr(MergeUpRayDataComps {
                trace_transmittance: tr.1,
                trace_color: tr.0,
            }),
        );
    }));

    let merge_up = DEVICE.create_kernel::<fn(u32, u32)>(&track!(|lv, face| {
        let grid = storage.grid_at(lv, face);
        let cell = dispatch_id().xy().cast_i32();
        let angle = dispatch_id().z;

        let data_index = grid.cell_index(cell, angle);
        let a = data_index != u32::MAX;
        lc_assert!(a);

        let last_grid = storage.grid_at(lv - 1, face);

        let last_angle = angle / 2;

        let offset_2 = 2 * angle.cast_i32() - grid.angle_resolution.cast_i32() + 1;
        let last_offset = 2 * last_angle.cast_i32() - last_grid.angle_resolution.cast_i32() + 1;

        let last_cell_0 = Vec2::expr(cell.x, cell.y * 2);
        let last_cell_1 = Vec2::expr(cell.x + offset_2 - last_offset, cell.y * 2 + 1);

        let tr = over2(
            storage.load_grid_up(last_grid, last_cell_0, last_angle),
            storage.load_grid_up(last_grid, last_cell_1, last_angle),
        );

        storage.merge_up_data.write(
            data_index,
            MergeUpRayData::from_comps_expr(MergeUpRayDataComps {
                trace_transmittance: tr.1,
                trace_color: tr.0,
            }),
        );
    }));

    let merge =
        DEVICE.create_kernel::<fn(u32, u32, Vec2<f32>, f32)>(&track!(|lv, face, pos, r| {
            let grid = storage.grid_at(lv, face);
            let cell = dispatch_id().xy().cast_i32();
            let angle = dispatch_id().z;

            let data_index = grid.cell_index(cell, angle);
            let a = data_index != u32::MAX;
            lc_assert!(a);

            let next_grid = storage.grid_at(lv + 1, face);

            let total_size = grid.angle_size(angle);
            let upper_size = next_grid.angle_size(angle * 2);
            let lower_size = next_grid.angle_size(angle * 2 + 1);

            let a = (upper_size + lower_size - total_size).abs() < 0.001;
            lc_assert!(a);

            let offset = 2 * angle.cast_i32() - grid.angle_resolution.cast_i32() + 1;
            let offset_0 = 2 * angle.cast_i32() - grid.angle_resolution.cast_i32();
            let offset_1 = 2 * angle.cast_i32() - grid.angle_resolution.cast_i32() + 2;

            let radiance = if cell.y % 2 == 0 {
                let c = Vec2::expr(cell.x + offset * 2, cell.y / 2 + 1);

                let tr_0 = trace_between(
                    grid.to_world(cell.cast_f32()),
                    next_grid.to_world(c.cast_f32() - 2.0 * Vec2::<f32>::x().expr()),
                    (pos, r),
                );
                let tr_1 = trace_between(
                    grid.to_world(cell.cast_f32()),
                    next_grid.to_world(c.cast_f32() + 2.0 * Vec2::<f32>::x().expr()),
                    (pos, r),
                );

                // let tr_0 = trace_frustrum(
                //     grid.to_world(cell.cast_f32()),
                //     next_grid.to_world(c.cast_f32()),
                //     grid.to_world(cell.cast_f32() - Vec2::x()),
                //     next_grid.to_world(c.cast_f32() - 4.0 * Vec2::<f32>::x().expr()),
                //     (pos, r),
                // );
                // let tr_1 = trace_frustrum(
                //     grid.to_world(cell.cast_f32()),
                //     next_grid.to_world(c.cast_f32()),
                //     grid.to_world(cell.cast_f32() + Vec2::x()),
                //     next_grid.to_world(c.cast_f32() + 4.0 * Vec2::<f32>::x().expr()),
                //     (pos, r),
                // );

                // Grids are directly overlapping.
                ((storage.load_grid(next_grid, Vec2::expr(cell.x, cell.y / 2), angle * 2)
                    + over(
                        tr_0,
                        storage.load_grid(
                            next_grid,
                            Vec2::expr(cell.x + offset_0 * 2, cell.y / 2 + 1),
                            angle * 2,
                        ),
                    ))
                    / 2.0
                    * upper_size
                    + (storage.load_grid(next_grid, Vec2::expr(cell.x, cell.y / 2), angle * 2 + 1)
                        + over(
                            tr_1,
                            storage.load_grid(
                                next_grid,
                                Vec2::expr(cell.x + offset_1 * 2, cell.y / 2 + 1),
                                angle * 2 + 1,
                            ),
                        ))
                        / 2.0
                        * lower_size)
                    / (upper_size + lower_size)
            } else {
                let c = Vec2::expr(cell.x + offset, cell.y / 2 + 1);
                let c_0 = Vec2::expr(cell.x + offset_0, cell.y / 2 + 1);
                let c_1 = Vec2::expr(cell.x + offset_1, cell.y / 2 + 1);

                let tr_0 = trace_between(
                    grid.to_world(cell.cast_f32()),
                    next_grid.to_world(c_0.cast_f32()),
                    (pos, r),
                );

                let tr_1 = trace_between(
                    grid.to_world(cell.cast_f32()),
                    next_grid.to_world(c_1.cast_f32()),
                    (pos, r),
                );

                // let tr_0 = trace_frustrum(
                //     grid.to_world(cell.cast_f32()),
                //     next_grid.to_world(c.cast_f32()),
                //     grid.to_world(cell.cast_f32() - Vec2::x()),
                //     next_grid.to_world(c.cast_f32() - 2.0 * Vec2::<f32>::x().expr()),
                //     (pos, r),
                // );
                // let tr_1 = trace_frustrum(
                //     grid.to_world(cell.cast_f32()),
                //     next_grid.to_world(c.cast_f32()),
                //     grid.to_world(cell.cast_f32() + Vec2::x()),
                //     next_grid.to_world(c.cast_f32() + 2.0 * Vec2::<f32>::x().expr()),
                //     (pos, r),
                // );

                // let tr = storage.load_grid_up(grid, cell, angle);
                // let tr = trace(
                //     dir,
                //     grid.to_world(cell.cast_f32()),
                //     Vec2::expr(0.0, grid.ray_len(dir)),
                // );

                let incoming_radiance = over(tr_0, storage.load_grid(next_grid, c_0, angle * 2))
                    * upper_size
                    + over(tr_1, storage.load_grid(next_grid, c_1, angle * 2 + 1)) * lower_size;

                incoming_radiance / (upper_size + lower_size)
            };

            storage.data.write(
                data_index,
                RayData::from_comps_expr(RayDataComps { color: radiance }),
            );
        }));

    let final_display = DEVICE.create_kernel::<fn()>(&track!(|| {
        let pos = dispatch_id().xy().cast_f32() + corner;
        let radiance = 0.0_f32.var();
        for i in 0_u32..4_u32 {
            let grid = storage.grid_at(0.expr(), i);
            let cell = grid.from_world(pos + grid.axis_y);
            // let a = (cell - cell.round()).abs().reduce_max() <= 0.00001;
            // lc_assert!(a);
            let cell = cell.round().cast_i32();
            *radiance += storage.load_grid(grid, cell, 0_u32.expr());
        }
        app.display()
            .write(dispatch_id().xy(), Vec3::splat_expr(radiance / 4.0));
    }));

    let draw_line_t = DEVICE.create_kernel::<fn(Vec2<f32>, Vec2<f32>, Vec3<f32>)>(&track!(
        |start, end, color| {
            let t = dispatch_id().x.cast_f32() / dispatch_size().x.cast_f32();
            let pos = start * (1.0 - t) + end * t;
            app.display().write(pos.cast_u32(), color);
        }
    ));
    let draw_line = |start: FVec2, end: FVec2, color: Vec3<f32>, spacing: f32| {
        draw_line_t.dispatch(
            [((start - end).length() * spacing).ceil() as u32, 1, 1],
            &Vec2::from(start),
            &Vec2::from(end),
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

    let mut is_tracing = false;
    let mut display_frustrums = false;

    app.run(|rt, _scope| {
        if rt.just_pressed_key(KeyCode::Backslash) {
            is_tracing = !is_tracing;
        }

        // 8.0: 1018 = 1024 - 6
        // 4.0: 1022 = 1024 - 2

        let r = 4.0; // (rt.tick as f32 / 60.0).sin() * 12.0 + 14.0;
                     // Vec2::new(1144.0, 1022.0);
                     // 1019.25, 1606.28

        if is_tracing {
            trace_kernel.dispatch([grid_size[0], grid_size[1], 1], &light_pos, &r);
        } else if true {
            let mut time = 0.0;
            // for dir in 0..4 {
            //     init_merge_up.dispatch([BASE_SIZE.x, BASE_SIZE.y, 1], &dir, &r);
            //     for i in 1..num_cascades {
            //         merge_up.dispatch([BASE_SIZE.x, BASE_SIZE.y >> i, 1 << i], &i, &dir);
            //     }
            // }

            for dir in 0..4 {
                let commands = (0..num_cascades)
                    .rev()
                    .map(|i| {
                        merge.dispatch_async(
                            [BASE_SIZE.x, BASE_SIZE.y >> i, 1 << i],
                            &i,
                            &dir,
                            &light_pos,
                            &r,
                        )
                    })
                    .collect::<Vec<_>>()
                    .chain();
                let timings = commands.execute_timed();
                time += timings.iter().map(|(_, t)| t).sum::<f32>();
            }
            final_display.dispatch([grid_size[0], grid_size[1], 1]);
            if rt.tick % 60 == 0 {
                println!("Time: {:?}ms", time);
            }
        }

        if rt.just_pressed_key(KeyCode::KeyF) {
            display_frustrums = !display_frustrums;
        }

        if rt.just_pressed_button(MouseButton::Right) {
            println!("{:?}", rt.cursor_position);
        }

        if rt.pressed_button(MouseButton::Left) {
            light_pos = rt.cursor_position;
        }

        rt.log_fps();
    });
}

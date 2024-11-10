use std::marker::PhantomData;

use keter::lang::types::vector::Vec2;
use keter::prelude::*;
use keter::runtime::KernelParameter;

use crate::color::{Fluence, MergeFluence, Radiance};
use crate::{Grid, Probe};

pub trait Tracer<F: MergeFluence> {
    type Params: KernelParameter;
    /// Traces the upper and lower frustrums for a given probe.
    ///
    fn trace(
        &self,
        params: &Self::Params,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> [Expr<Fluence<F>>; 2];
}

pub trait WorldTracer<F: MergeFluence> {
    type Params: KernelParameter;

    fn trace(
        &self,
        params: &Self::Params,
        start: Expr<Vec2<f32>>,
        end: Expr<Vec2<f32>>,
    ) -> Expr<Fluence<F>>;
}

pub struct WorldMapper<F: MergeFluence, T> {
    pub tracer: T,
    pub world_origin: Vec2<f32>,
    pub world_size: Vec2<f32>,
    pub _marker: PhantomData<F>,
}
impl<F: MergeFluence, T> WorldMapper<F, T> {
    #[tracked]
    pub fn to_world(&self, grid: Expr<Grid>, cell: Expr<Vec2<u32>>) -> Expr<Vec2<f32>> {
        // oob handling lol
        cell.cast_i32().cast_f32() / grid.size.cast_f32() * self.world_size + self.world_origin
    }
}

impl<F: MergeFluence, T: WorldTracer<F>> Tracer<F> for WorldMapper<F, T> {
    type Params = T::Params;
    #[tracked]
    fn trace(
        &self,
        params: &Self::Params,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> [Expr<Fluence<F>>; 2] {
        let start = self.to_world(grid, probe.cell);
        let offset = grid.offset(probe.dir);
        let next_grid = grid.next();
        let spacing = (self.world_size.y / grid.size.y.cast::<f32>())
            / (self.world_size.x / grid.size.x.cast::<f32>());
        let lower_size = next_grid.angle_size(spacing, probe.dir * 2);
        let upper_size = next_grid.angle_size(spacing, probe.dir * 2 + 1);

        if probe.cell.y % 2 == 0 {
            let end = probe.cell + Vec2::expr(offset * 2, 2);
            [
                self.tracer
                    .trace(params, start, self.to_world(grid, end - 2 * Vec2::x()))
                    .restrict_angle(lower_size),
                self.tracer
                    .trace(params, start, self.to_world(grid, end + 2 * Vec2::x()))
                    .restrict_angle(upper_size),
            ]
        } else {
            let end = probe.cell + Vec2::expr(offset, 1);
            [
                self.tracer
                    .trace(params, start, self.to_world(grid, end - Vec2::x()))
                    .restrict_angle(lower_size),
                self.tracer
                    .trace(params, start, self.to_world(grid, end + Vec2::x()))
                    .restrict_angle(upper_size),
            ]
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle<R> {
    pub center: Vec2<f32>,
    pub radius: f32,
    pub color: R,
}

pub struct AnalyticTracer<F: MergeFluence> {
    pub circles: Vec<Circle<F::Radiance>>,
}

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

/*
impl WorldTracer<BinaryF32> for AnalyticTracer<BinaryF32> {
    type Params = ();
    #[tracked]
    fn trace(
        &self,
        _params: &Self::Params,
        start: Expr<Vec2<f32>>,
        end: Expr<Vec2<f32>>,
    ) -> Expr<Fluence<BinaryF32>> {
        let end_t = (end - start).length();
        let dir = (end - start).normalize();

        let best_t = end_t.var();
        let best_color = f32::black().var();
        for Circle {
            center,
            radius,
            color,
        } in self.circles.iter()
        {
            let (min_t, max_t, penetration, hit) =
                intersect_circle(start - center, dir, radius.expr());
            if hit && max_t > 0.0 && min_t < best_t {
                *best_t = min_t;
                *best_color = f32::scale(color.expr(), keter::min(penetration / 2.0, 1.0));
            }
        }
        // if best_t < end_t {
        Fluence::expr(1.0_f32.expr(), false.expr())
        // } else {
        //     Fluence::empty().expr()
        // }
    }
}
 */

impl<F: MergeFluence<Transmittance = bool>> WorldTracer<F> for AnalyticTracer<F> {
    type Params = ();
    #[tracked]
    fn trace(
        &self,
        _params: &Self::Params,
        start: Expr<Vec2<f32>>,
        end: Expr<Vec2<f32>>,
    ) -> Expr<Fluence<F>> {
        let end_t = (end - start).length();
        let dir = (end - start).normalize();

        let best_t = end_t.var();
        let best_color = F::Radiance::black().var();
        for Circle {
            center,
            radius,
            color,
        } in self.circles.iter()
        {
            let (min_t, max_t, penetration, hit) =
                intersect_circle(start - center, dir, radius.expr());
            if hit && max_t > 0.0 && min_t < best_t {
                *best_t = min_t;
                *best_color = F::Radiance::scale(color.expr(), keter::min(penetration / 2.0, 1.0));
            }
        }
        if best_t < end_t {
            Fluence::solid_expr(**best_color)
        } else {
            Fluence::empty().expr()
        }
    }
}

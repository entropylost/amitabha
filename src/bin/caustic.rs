use std::f32::consts::TAU;

use amitabha::fluence::Fluence;
use amitabha::render::{HRCRenderer, HRCSettings};
use amitabha::trace::{AnalyticTracer, Circle, WorldTracer};
use amitabha::utils::pcgf_host;
use amitabha::{color, fluence};
use keter::graph::profile::Profiler;
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::App;

const DISPLAY_SIZE: u32 = 1024;
type F = fluence::RgbF16;
type C = color::RgbF16;

#[tracked]
fn wave(x: Expr<f32>, t: Expr<f32>) -> Expr<f32> {
    let median_wavelength = 500.0_f32;
    let amplitude = 40.0_f32;
    let mut seed = 10;
    let mut rand = || {
        seed += 1;
        pcgf_host(seed)
    };
    let mut height = 0.0_f32.expr();
    #[allow(unused_parens)]
    for _ in (0..1) {
        let wavelength = median_wavelength * 2.0_f32.powf(2.0_f32 * rand() - 1.0_f32);
        let phase = rand();
        height = height
            + (TAU * x / wavelength - TAU * phase + (1000.0 / wavelength).sqrt() * t).sin()
                * amplitude
                * wavelength
                / median_wavelength;
    }
    height
}

pub struct CausticTracer {
    height: Box<dyn Fn(Expr<f32>) -> Expr<f32>>,
    min_height: f32,
    max_slope: f32,
    refr_index: f32,
    tracer: AnalyticTracer<F>,
}

impl WorldTracer<F> for CausticTracer {
    type Params = ();
    #[tracked]
    fn trace(
        &self,
        _params: &Self::Params,
        start: Expr<Vec2<f32>>,
        direction: Expr<Vec2<f32>>,
        length: Expr<f32>,
    ) -> Expr<Fluence<F>> {
        let start = Vec2::expr(start.x, 1024.0 - start.y);
        let direction = Vec2::expr(direction.x, -direction.y);

        if start.y >= (self.height)(start.x) {
            self.tracer.trace(&(), start, direction, length)
        } else if direction.y < 0.0 {
            Fluence::dark(Vec3::splat(f16::ZERO)).expr()
        } else {
            let t = 0.0_f32.var();
            if self.min_height > start.y {
                *t = (self.min_height - start.y) / direction.y;
            }
            while t < length {
                let pos = start + direction * t;
                let height = (self.height)(pos.x) - pos.y;
                if height < 1.0 {
                    break;
                }
                *t += height / (self.max_slope * direction.x.abs() + direction.y);
            }
            if t >= length {
                Fluence::empty().expr()
            } else {
                let pos = start + direction * t;
                let slope = (self.height)(pos.x + 0.001) - (self.height)(pos.x - 0.001);
                let slope = slope / 0.002;
                let normal = Vec2::expr(-slope, 1.0).normalize();
                let angle = (1.0 - normal.dot(direction).sqr()).sqrt() * self.refr_index; // sin theta
                if angle.abs() >= 1.0 {
                    Fluence::dark(Vec3::splat(f16::ZERO)).expr()
                } else {
                    let tangent = Vec2::expr(normal.y, -normal.x);
                    let sign = direction.dot(tangent).signum();
                    let dir = angle * sign * tangent + (1.0 - angle.sqr()).sqrt() * normal;
                    let fluence = self.tracer.trace(&(), pos, dir, 10000.0_f32.expr());
                    Fluence::solid_expr(fluence.radiance)
                }
            }
        }
    }
}

fn main() {
    let app = App::new("Amitabha", [DISPLAY_SIZE; 2])
        .scale(2048 / DISPLAY_SIZE)
        .agx()
        .init();
    let time = Singleton::new(0.0_f32);
    let view = time.0.view(..);
    let world = CausticTracer {
        height: Box::new(track!(move |x| wave(x, time.read()) + 500.0)),
        min_height: 400.0,
        max_slope: 5.0,
        refr_index: 1.33,
        tracer: AnalyticTracer {
            buffer: DEVICE.create_buffer_from_slice(&[Circle {
                center: Vec2::new(500.0, 1000.0),
                radius: 200.0,
                color: Vec3::splat(f16::from_f32(0.4)),
            }]),
        },
    };

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
        if rt.tick == 0 {
            rt.begin_recording(None, false);
        }
        if rt.tick == 360 {
            rt.finish_recording();
            println!("Done");
        }
        view.copy_from(&[rt.tick as f32 * 0.016]);
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
            // profiler.print(&["Merge Up", "Merge Down", "Finish", "Draw"], true);
            profiler.reset();
        }
    });
}

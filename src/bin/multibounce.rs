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

pub enum Brush {
    Rect(f32, f32),
    Circle(f32),
}
#[derive(Debug, Clone, Copy)]
pub struct SceneColor {
    emission: Vec3<f32>,
    opacity: Vec3<f32>,
    diffuse: Vec3<f32>,
}
impl SceneColor {
    fn new(emission: Vec3<f32>, opacity: Vec3<f32>, diffuse: Vec3<f32>) -> Self {
        Self {
            emission,
            opacity,
            diffuse,
        }
    }
    fn dark(opacity: Vec3<f32>, diffuse: Vec3<f32>) -> Self {
        Self {
            emission: Vec3::splat(0.0),
            diffuse,
            opacity,
        }
    }
    fn bright(emission: Vec3<f32>) -> Self {
        Self {
            emission,
            diffuse: Vec3::splat(0.0),
            opacity: Vec3::splat(f32::INFINITY),
        }
    }
    fn solid(diffuse: Vec3<f32>) -> Self {
        Self {
            emission: Vec3::splat(0.0),
            diffuse,
            opacity: Vec3::splat(0.5),
        }
    }
    fn diffuse(&self) -> Vec3<f16> {
        Vec3::new(
            f16::from_f32(self.diffuse.x),
            f16::from_f32(self.diffuse.y),
            f16::from_f32(self.diffuse.z),
        )
    }
}
impl From<SceneColor> for Color<C> {
    fn from(
        SceneColor {
            emission, opacity, ..
        }: SceneColor,
    ) -> Self {
        Self::new(
            Vec3::new(
                f16::from_f32(emission.x),
                f16::from_f32(emission.y),
                f16::from_f32(emission.z),
            ),
            Vec3::new(
                f16::from_f32(opacity.x),
                f16::from_f32(opacity.y),
                f16::from_f32(opacity.z),
            ),
        )
    }
}

pub struct Draw {
    pub brush: Brush,
    pub center: Vec2<f32>,
    pub color: SceneColor,
}

pub struct Scene {
    pub draws: Vec<Draw>,
}
impl Scene {
    fn cornell() -> Self {
        Self {
            draws: vec![
                Draw {
                    brush: Brush::Rect(256.0, 10.0),
                    center: Vec2::new(256.0, 512.0),
                    color: SceneColor::solid(Vec3::splat(1.0)),
                },
                Draw {
                    brush: Brush::Rect(256.0, 10.0),
                    center: Vec2::new(256.0, 0.0),
                    color: SceneColor::solid(Vec3::splat(1.0)),
                },
                Draw {
                    brush: Brush::Rect(10.0, 256.0),
                    center: Vec2::new(0.0, 256.0),
                    color: SceneColor::solid(Vec3::new(1.0, 0.0, 0.0)),
                },
                Draw {
                    brush: Brush::Rect(10.0, 256.0),
                    center: Vec2::new(512.0, 256.0),
                    color: SceneColor::solid(Vec3::new(0.0, 0.0, 1.0)),
                },
                Draw {
                    brush: Brush::Rect(34.0, 10.0),
                    center: Vec2::new(256.0, 0.0),
                    color: SceneColor::bright(Vec3::splat(0.0)),
                },
                Draw {
                    brush: Brush::Rect(32.0, 10.0),
                    center: Vec2::new(256.0, 0.0),
                    color: SceneColor::bright(Vec3::splat(3.0)),
                },
                Draw {
                    brush: Brush::Circle(96.0),
                    center: Vec2::new(256.0 - 96.0, 512.0 - 96.0 - 10.0),
                    color: SceneColor::solid(Vec3::new(0.0, 0.0, 0.0)),
                },
                Draw {
                    brush: Brush::Rect(48.0, 96.0),
                    center: Vec2::new(384.0, 416.0 - 10.0),
                    color: SceneColor::dark(Vec3::new(0.03, 0.02, 0.03), Vec3::splat(0.8)),
                },
            ],
        }
    }
}

fn main() {
    let app = App::new("Amitabha", [DISPLAY_SIZE; 2])
        .scale(2048 / DISPLAY_SIZE)
        .agx()
        .init();
    let world = VoxelTracer::<C>::new(Vec2::new(DISPLAY_SIZE, DISPLAY_SIZE));
    let diffuse_tex =
        DEVICE.create_tex2d::<Vec3<f16>>(PixelStorage::Float4, DISPLAY_SIZE, DISPLAY_SIZE, 1);

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

        let diffuse = diffuse_tex.read(cell);
        if (world.read(cell).opacity < f16::from_f32(200.0)).any() {
            let emission = diffuse * radiance;
            world.write_emission(cell, emission);
        }

        app.display().write(cell, radiance.cast_f32());
    }));

    let rect_brush = DEVICE.create_kernel::<fn(Vec2<f32>, Vec2<f32>, Color<C>, Vec3<f16>)>(
        &track!(|center, size, color, diffuse| {
            let pos = dispatch_id().xy();
            if ((pos.cast_f32() + 0.5 - center).abs() < size).all() {
                world.write(pos, color);
                diffuse_tex.write(pos, diffuse);
            }
        }),
    );
    let circle_brush = DEVICE.create_kernel::<fn(Vec2<f32>, f32, Color<C>, Vec3<f16>)>(&track!(
        |center, radius, color, diffuse| {
            let pos = dispatch_id().xy();
            if (pos.cast_f32() + 0.5 - center).length() < radius {
                world.write(pos, color);
                diffuse_tex.write(pos, diffuse);
            }
        }
    ));

    let scene = Scene::cornell();
    for Draw {
        brush,
        center,
        color,
    } in scene.draws
    {
        match brush {
            Brush::Rect(width, height) => {
                rect_brush.dispatch(
                    [world.size.x, world.size.y, 1],
                    &center,
                    &Vec2::new(width, height),
                    &Color::from(color),
                    &color.diffuse(),
                );
            }
            Brush::Circle(radius) => {
                circle_brush.dispatch(
                    [world.size.x, world.size.y, 1],
                    &center,
                    &radius,
                    &Color::from(color),
                    &color.diffuse(),
                );
            }
        }
    }

    let mut profiler = Profiler::new();

    app.run(|rt| {
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

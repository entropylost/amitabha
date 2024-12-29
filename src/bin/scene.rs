use amitabha::color::{Color, RgbF16};
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use palette::{FromColor, LinSrgb, Oklch};

const DISPLAY_SIZE: f32 = 512.0;

pub enum Brush {
    Rect(f32, f32),
    Circle(f32),
}
pub struct SceneColor {
    emission: Vec3<f32>,
    opacity: Vec3<f32>,
}
impl SceneColor {
    fn new(emission: Vec3<f32>, opacity: Vec3<f32>) -> Self {
        Self { emission, opacity }
    }
    fn dark(opacity: Vec3<f32>) -> Self {
        Self {
            emission: Vec3::splat(0.0),
            opacity,
        }
    }
    fn solid(emission: Vec3<f32>) -> Self {
        Self {
            emission,
            opacity: Vec3::splat(f32::INFINITY),
        }
    }
}
impl From<SceneColor> for Color<RgbF16> {
    fn from(SceneColor { emission, opacity }: SceneColor) -> Self {
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
#[allow(dead_code)]
impl Scene {
    pub fn penumbra_example(width: f32, height: f32) -> Self {
        let center = Vec2::splat(DISPLAY_SIZE / 2.0);
        Self {
            draws: vec![
                Draw {
                    brush: Brush::Rect(1.0, height),
                    center: center + Vec2::y() * height,
                    color: SceneColor::solid(Vec3::splat(0.0)),
                },
                Draw {
                    brush: Brush::Rect(1.0, height),
                    center: center - Vec2::x() * width,
                    color: SceneColor::solid(Vec3::splat(5.0)),
                },
            ],
        }
    }
    pub fn sunflower() -> Self {
        let spacing = 15.0;
        let center = Vec2::splat(DISPLAY_SIZE / 2.0);
        let mut draws = vec![];
        for i in 1..200 {
            let r = spacing * (i as f32).sqrt();
            let angle = i as f32 * 137.508_f32.to_radians();
            let pos = center + Vec2::new(angle.cos(), angle.sin()) * r;

            let brightness = (1.0 - r / 150.0).max(0.0);
            let color = Oklch::new(brightness, 0.15, angle.to_degrees());
            let color = LinSrgb::from_color(color);

            draws.push(Draw {
                brush: Brush::Circle(5.0),
                center: pos,
                color: SceneColor::new(
                    Vec3::new(
                        color.red.max(0.0),
                        color.green.max(0.0),
                        color.blue.max(0.0),
                    ),
                    Vec3::splat(0.7),
                ),
            });
        }
        Self { draws }
    }
    pub fn sunflower2() -> Self {
        let spacing = 15.0;
        let center = Vec2::splat(DISPLAY_SIZE / 2.0);
        let mut draws = vec![Draw {
            brush: Brush::Circle(15.0),
            center,
            color: SceneColor::new(Vec3::splat(5.0), Vec3::splat(0.5)),
        }];
        for i in 6..200 {
            let r = spacing * (i as f32).sqrt();
            let angle = i as f32 * 137.508_f32.to_radians();
            let pos = center + Vec2::new(angle.cos(), angle.sin()) * r;

            let brightness = (r / 200.0).max(0.5);
            let color = Oklch::new(brightness, 0.15, angle.to_degrees());
            let color = LinSrgb::from_color(color);

            draws.push(Draw {
                brush: Brush::Circle(5.0),
                center: pos,
                color: SceneColor::dark(Vec3::new(
                    color.red.max(0.0),
                    color.green.max(0.0),
                    color.blue.max(0.0),
                )),
            });
        }
        Self { draws }
    }
    pub fn sunflower3() -> Self {
        let spacing = 15.0;
        let center = Vec2::splat(DISPLAY_SIZE / 2.0);
        let mut draws = vec![];
        for i in 1..200 {
            let r = spacing * (i as f32).sqrt();
            let angle = i as f32 * 137.508_f32.to_radians();
            let pos = center + Vec2::new(angle.cos(), angle.sin()) * r;

            let brightness = (1.0 - r / 150.0).max(0.0);
            let opacity = (r / 200.0).max(0.5);
            let color = Oklch::new(opacity, 0.15, angle.to_degrees());
            let color = LinSrgb::from_color(color);

            draws.push(Draw {
                brush: Brush::Circle(5.0),
                center: pos,
                color: SceneColor::new(
                    Vec3::splat(brightness),
                    Vec3::new(
                        color.red.max(0.0),
                        color.green.max(0.0),
                        color.blue.max(0.0),
                    ),
                ),
            });
        }
        Self { draws }
    }
    pub fn sunflower4() -> Self {
        let spacing = 30.0;
        let center = Vec2::splat(1024.0 / 2.0);
        let mut draws = vec![];
        for i in 1..100 {
            let r = spacing * (i as f32).sqrt();
            let angle = i as f32 * 137.508_f32.to_radians();
            let pos = center + Vec2::new(angle.cos(), angle.sin()) * r;

            let brightness = (1.0 - r / 150.0).max(0.0);
            let color = Oklch::new(brightness, 0.15, angle.to_degrees());
            let color = LinSrgb::from_color(color);

            draws.push(Draw {
                brush: Brush::Circle(10.0),
                center: pos,
                color: SceneColor::new(
                    Vec3::new(
                        5.0 * color.red.max(0.0),
                        5.0 * color.green.max(0.0),
                        5.0 * color.blue.max(0.0),
                    ),
                    Vec3::splat(0.7),
                ),
            });
        }
        Self { draws }
    }

    pub fn opacitytest() -> Self {
        let mut draws = vec![Draw {
            brush: Brush::Circle(15.0),
            center: Vec2::new(50.0, 256.0),
            color: SceneColor::new(Vec3::splat(5.0), Vec3::splat(0.5)),
        }];
        for i in 0..50 {
            let opacity = 0.01 * i as f32;
            draws.push(Draw {
                brush: Brush::Rect(5.0, 2.0),
                center: Vec2::new(250.0, 200.0 + i as f32 * 2.0),
                color: SceneColor::dark(Vec3::splat(opacity)),
            })
        }
        Self { draws }
    }
    pub fn simple() -> Self {
        Self {
            draws: vec![
                Draw {
                    brush: Brush::Circle(5.0),
                    center: Vec2::new(30.0, 256.0),
                    color: SceneColor::new(Vec3::splat(5.0), Vec3::splat(0.5)),
                },
                Draw {
                    brush: Brush::Circle(7.0),
                    center: Vec2::new(300.0, 256.0),
                    color: SceneColor::dark(Vec3::splat(0.5)),
                },
            ],
        }
    }
    pub fn pinhole() -> Self {
        let mut draws = vec![
            Draw {
                brush: Brush::Rect(1.0, 512.0),
                center: Vec2::new(512.0, -5.0),
                color: SceneColor::solid(Vec3::splat(0.0)),
            },
            Draw {
                brush: Brush::Rect(1.0, 512.0),
                center: Vec2::new(512.0, 1024.0 + 5.0),
                color: SceneColor::solid(Vec3::splat(0.0)),
            },
        ];
        for i in -3..=3 {
            let color = Oklch::new(0.5, 0.15, (1.618033988 * 360.0 * i as f64) % 360.0);
            let color = LinSrgb::from_color(color);
            draws.push(Draw {
                brush: Brush::Rect(5.0, 20.0),
                center: Vec2::new(1000.0, i as f32 * 40.0 + 512.0),
                color: SceneColor::solid(Vec3::new(
                    50.0 * color.red.max(0.0) as f32,
                    50.0 * color.green.max(0.0) as f32,
                    50.0 * color.blue.max(0.0) as f32,
                )),
            })
        }
        Self { draws }
    }
}

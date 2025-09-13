use std::f32::consts::TAU;

use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use palette::{FromColor, LinSrgb, Oklch};

use crate::color::{Color, RgbF16};

#[derive(Debug, Clone, Copy)]
pub enum Brush {
    Rect(f32, f32),
    Circle(f32),
}
#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy)]
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
    pub fn penumbra_example(display_size: f32, width: f32, height: f32) -> Self {
        let center = Vec2::splat(display_size / 2.0);
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
    // for 1024x1024
    pub fn sunflower() -> Self {
        let spacing = 15.0;
        let center = Vec2::splat(512.0);
        let mut draws = vec![];
        for i in 1..200 {
            let r = spacing * (i as f32).sqrt();
            let angle = i as f32 * 137.508_f32.to_radians();
            let pos = center + angle.direction() * r;

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
    // for 1024x1024
    pub fn sunflower2() -> Self {
        let spacing = 15.0;
        let center = Vec2::splat(512.0);
        let mut draws = vec![Draw {
            brush: Brush::Circle(15.0),
            center,
            color: SceneColor::new(Vec3::splat(5.0), Vec3::splat(0.5)),
        }];
        for i in 6..200 {
            let r = spacing * (i as f32).sqrt();
            let angle = i as f32 * 137.508_f32.to_radians();
            let pos = center + angle.direction() * r;

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
    // for 1024x1024
    pub fn sunflower3() -> Self {
        let spacing = 15.0;
        let center = Vec2::splat(512.0);
        let mut draws = vec![];
        for i in 1..200 {
            let r = spacing * (i as f32).sqrt();
            let angle = i as f32 * 137.508_f32.to_radians();
            let pos = center + angle.direction() * r;

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
    // for 1024x1024
    pub fn sunflower4() -> Self {
        let spacing = 30.0;
        let center = Vec2::splat(512.0);
        let mut draws = vec![];
        for i in 1..100 {
            let r = spacing * (i as f32).sqrt();
            let angle = i as f32 * 137.508_f32.to_radians();
            let pos = center + angle.direction() * r;

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
    // for 512x512
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
    pub fn top() -> Self {
        Self {
            draws: vec![Draw {
                brush: Brush::Rect(10000.0, 10.0),
                center: Vec2::new(0.0, 0.0),
                color: SceneColor::new(Vec3::new(5.0, 5.0, 5.0), Vec3::splat(0.5)),
            }],
        }
    }
    // for 512x512
    pub fn simple() -> Self {
        Self {
            draws: vec![
                Draw {
                    brush: Brush::Circle(5.0),
                    center: Vec2::new(30.0, 256.0),
                    color: SceneColor::new(Vec3::new(10.0, 10.5, 11.0), Vec3::splat(0.5)),
                },
                Draw {
                    brush: Brush::Circle(7.0),
                    center: Vec2::new(300.0, 256.0),
                    color: SceneColor::dark(Vec3::splat(10000.0)),
                },
            ],
        }
    }
    // for 2048x2048
    pub fn point() -> Self {
        Self {
            draws: vec![Draw {
                brush: Brush::Circle(8.1),
                center: Vec2::new(1024.0, 1024.0),
                color: SceneColor::new(Vec3::splat(10.0 / 8.0), Vec3::splat(10.0)),
            }],
        }
    }
    // for 1024x1024
    pub fn pinhole(t: u32) -> Self {
        let l = (t as f32 / 200.0 * TAU).sin() * 200.0;
        let mut draws = vec![
            Draw {
                brush: Brush::Rect(1025.0, 1025.0),
                center: Vec2::new(0.0, 0.0),
                color: SceneColor::dark(Vec3::splat(0.0)),
            },
            Draw {
                brush: Brush::Rect(1.0, 512.0 + l),
                center: Vec2::new(512.0, -5.0),
                color: SceneColor::solid(Vec3::splat(0.0)),
            },
            Draw {
                brush: Brush::Rect(1.0, 512.0 - l),
                center: Vec2::new(512.0, 1024.0 + 5.0),
                color: SceneColor::solid(Vec3::splat(0.0)),
            },
        ];
        for i in -3..=3 {
            let color = Oklch::new(0.5, 0.15, (1.618033988 * 360.0 * i as f64) % 360.0);
            let color = LinSrgb::from_color(color);
            draws.push(Draw {
                brush: Brush::Rect(5.0, 20.0),
                center: Vec2::new(1024.0 - 1000.0, -i as f32 * 40.0 + 512.0),
                color: SceneColor::solid(Vec3::new(
                    50.0 * color.red.max(0.0) as f32,
                    50.0 * color.green.max(0.0) as f32,
                    50.0 * color.blue.max(0.0) as f32,
                )),
            })
        }
        Self { draws }
    }
    // for 512x512
    pub fn volume() -> Self {
        Self {
            draws: vec![
                Draw {
                    brush: Brush::Circle(30.0),
                    center: Vec2::new(256.0, 50.0),
                    color: SceneColor::solid(Vec3::new(2.3, 0.5, 1.2)),
                },
                Draw {
                    brush: Brush::Rect(100.0, 100.0),
                    center: Vec2::new(256.0, 256.0),
                    color: SceneColor::dark(Vec3::new(0.02, 0.01, 0.01)),
                },
                Draw {
                    brush: Brush::Circle(15.0),
                    center: Vec2::new(180.0, 256.0),
                    color: SceneColor::solid(Vec3::new(0.8, 1.3, 0.8)),
                },
            ],
        }
    }
}

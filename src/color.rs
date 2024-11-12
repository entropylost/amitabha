use std::f32::consts::TAU;

use keter::lang::types::vector::Vec3;
use keter::prelude::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Value)]
pub struct Fluence<F: MergeFluence> {
    pub radiance: F::Radiance,
    pub transmittance: F::Transmittance,
}
impl<F: MergeFluence> Fluence<F> {
    #[tracked]
    pub fn expr(radiance: Expr<F::Radiance>, transmittance: Expr<F::Transmittance>) -> Expr<Self> {
        Fluence::from_comps_expr(FluenceComps {
            radiance,
            transmittance,
        })
    }
    pub fn empty() -> Self {
        Fluence {
            radiance: F::Radiance::black(),
            transmittance: F::Transmittance::transparent(),
        }
    }
    pub fn solid(radiance: F::Radiance) -> Self {
        Fluence {
            radiance,
            transmittance: F::Transmittance::opaque(),
        }
    }
    #[tracked]
    pub fn solid_expr(radiance: Expr<F::Radiance>) -> Expr<Self> {
        Fluence::expr(radiance, F::Transmittance::opaque().expr())
    }
}
impl<F: MergeFluence> FluenceExpr<F> {
    #[tracked]
    pub fn restrict_angle(self, angle: Expr<f32>) -> Expr<Fluence<F>> {
        Fluence::expr(
            F::Radiance::restrict_angle(self.radiance, angle),
            self.transmittance,
        )
    }
    #[tracked]
    pub fn opaque_if(self, x: Expr<bool>) -> Expr<Fluence<F>> {
        Fluence::expr(
            self.radiance,
            if x {
                F::Transmittance::opaque().expr()
            } else {
                self.transmittance
            },
        )
    }
}

pub trait MergeFluence: 'static + Copy {
    type Radiance: Radiance;
    type Transmittance: Transmittance;

    fn over(near: Expr<Fluence<Self>>, far: Expr<Fluence<Self>>) -> Expr<Fluence<Self>>;
    fn over_radiance(near: Expr<Fluence<Self>>, far: Expr<Self::Radiance>) -> Expr<Self::Radiance>;
}

pub trait Radiance: Value {
    // Use *0.5 instead of /2.0 since that actually matters.
    fn merge(a: Expr<Self>, b: Expr<Self>) -> Expr<Self>;
    fn blend(a: Expr<Self>, b: Expr<Self>) -> Expr<Self>;
    fn scale(this: Expr<Self>, scale: Expr<f32>) -> Expr<Self>;
    #[tracked]
    fn restrict_angle(this: Expr<Self>, angle: Expr<f32>) -> Expr<Self> {
        Self::scale(this, angle / TAU)
    }
    fn black() -> Self;

    fn debug_white() -> Self;
}

impl Radiance for f32 {
    #[tracked]
    fn merge(a: Expr<Self>, b: Expr<Self>) -> Expr<Self> {
        a + b
    }
    #[tracked]
    fn blend(a: Expr<Self>, b: Expr<Self>) -> Expr<Self> {
        (a + b) * 0.5
    }
    #[tracked]
    fn scale(this: Expr<Self>, scale: Expr<f32>) -> Expr<Self> {
        this * scale
    }
    #[tracked]
    fn black() -> Self {
        0.0
    }

    fn debug_white() -> Self {
        1.0
    }
}
impl Radiance for Vec3<f32> {
    #[tracked]
    fn merge(a: Expr<Self>, b: Expr<Self>) -> Expr<Self> {
        a + b
    }
    #[tracked]
    fn blend(a: Expr<Self>, b: Expr<Self>) -> Expr<Self> {
        (a + b) * 0.5
    }
    #[tracked]
    fn scale(this: Expr<Self>, scale: Expr<f32>) -> Expr<Self> {
        this * scale
    }
    #[tracked]
    fn black() -> Self {
        Vec3::splat(0.0)
    }

    fn debug_white() -> Self {
        Vec3::splat(1.0)
    }
}

pub trait Transmittance: Value {
    fn transparent() -> Self;
    fn opaque() -> Self;
}
impl Transmittance for bool {
    fn transparent() -> Self {
        true
    }
    fn opaque() -> Self {
        false
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BinaryF32;
impl MergeFluence for BinaryF32 {
    type Radiance = f32;
    type Transmittance = bool;
    #[tracked]
    fn over(near: Expr<Fluence<Self>>, far: Expr<Fluence<Self>>) -> Expr<Fluence<Self>> {
        Fluence::expr(
            near.transmittance.select(far.radiance, near.radiance),
            near.transmittance && far.transmittance,
        )
    }
    #[tracked]
    fn over_radiance(near: Expr<Fluence<Self>>, far: Expr<Self::Radiance>) -> Expr<Self::Radiance> {
        near.transmittance.select(far, near.radiance)
    }
}

use std::f32::consts::TAU;

use keter::lang::types::vector::Vec3;
use keter::prelude::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Value)]
pub struct Fluence<F: FluenceType> {
    pub radiance: F::Radiance,
    pub transmittance: F::Transmittance,
}
impl<F: FluenceType> Fluence<F> {
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
impl<F: FluenceType> FluenceExpr<F> {
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
impl<F: FluenceType> Fluence<F>
where
    F::Transmittance: PartialTransmittance,
{
    #[tracked]
    pub fn blend(a: Expr<Self>, b: Expr<Self>) -> Expr<Self> {
        Fluence::expr(
            F::Radiance::blend(a.radiance, b.radiance),
            F::Transmittance::blend(a.transmittance, b.transmittance),
        )
    }
    #[tracked]
    pub fn blend_n(v: &[Expr<Self>]) -> Expr<Self> {
        Fluence::expr(
            F::Radiance::blend_n(&v.iter().map(|x| x.radiance).collect::<Vec<_>>()),
            F::Transmittance::blend_n(&v.iter().map(|x| x.transmittance).collect::<Vec<_>>()),
        )
    }
}

pub trait FluenceType: 'static + Copy {
    type Radiance: Radiance;
    type Transmittance: Transmittance;

    fn over(near: Expr<Fluence<Self>>, far: Expr<Fluence<Self>>) -> Expr<Fluence<Self>>;
    fn over_radiance(near: Expr<Fluence<Self>>, far: Expr<Self::Radiance>) -> Expr<Self::Radiance>;
}

pub trait Radiance: Value {
    // Use *0.5 instead of /2.0 since that actually matters.
    fn merge(a: Expr<Self>, b: Expr<Self>) -> Expr<Self>;
    fn blend(a: Expr<Self>, b: Expr<Self>) -> Expr<Self>;
    fn blend_n(v: &[Expr<Self>]) -> Expr<Self>;
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
    fn blend_n(v: &[Expr<Self>]) -> Expr<Self> {
        let sum = v.iter().fold(0.0.expr(), |sum, &x| sum + x);
        sum * (1.0 / v.len() as f32)
    }
    #[tracked]
    fn scale(this: Expr<Self>, scale: Expr<f32>) -> Expr<Self> {
        this * scale
    }
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
    fn blend_n(v: &[Expr<Self>]) -> Expr<Self> {
        let sum = v.iter().fold(Vec3::splat_expr(0.0_f32), |sum, &x| sum + x);
        sum * (1.0 / v.len() as f32)
    }
    #[tracked]
    fn scale(this: Expr<Self>, scale: Expr<f32>) -> Expr<Self> {
        this * scale
    }
    fn black() -> Self {
        Vec3::splat(0.0)
    }

    fn debug_white() -> Self {
        Vec3::splat(1.0)
    }
}
impl Radiance for Vec3<f16> {
    #[tracked]
    fn merge(a: Expr<Self>, b: Expr<Self>) -> Expr<Self> {
        a + b
    }
    #[tracked]
    fn blend(a: Expr<Self>, b: Expr<Self>) -> Expr<Self> {
        (a + b) * f16::from_f32_const(0.5)
    }
    #[tracked]
    fn blend_n(v: &[Expr<Self>]) -> Expr<Self> {
        let sum = v
            .iter()
            .fold(Vec3::splat_expr(f16::ZERO), |sum, &x| sum + x);
        sum * f16::from_f32(1.0 / v.len() as f32)
    }
    #[tracked]
    fn scale(this: Expr<Self>, scale: Expr<f32>) -> Expr<Self> {
        this * scale.cast_f16()
    }
    fn black() -> Self {
        Vec3::splat(f16::ZERO)
    }

    fn debug_white() -> Self {
        Vec3::splat(f16::ONE)
    }
}

pub trait Transmittance: Value {
    fn transparent() -> Self;
    fn opaque() -> Self;
}
pub trait PartialTransmittance: Transmittance {
    fn blend(a: Expr<Self>, b: Expr<Self>) -> Expr<Self>;
    fn blend_n(v: &[Expr<Self>]) -> Expr<Self>;
}

impl Transmittance for bool {
    fn transparent() -> Self {
        true
    }
    fn opaque() -> Self {
        false
    }
}
impl Transmittance for f32 {
    fn transparent() -> Self {
        1.0
    }
    fn opaque() -> Self {
        0.0
    }
}
impl PartialTransmittance for f32 {
    #[tracked]
    fn blend(a: Expr<Self>, b: Expr<Self>) -> Expr<Self> {
        (a + b) * 0.5
    }
    #[tracked]
    fn blend_n(v: &[Expr<Self>]) -> Expr<Self> {
        let sum = v.iter().fold(0.0.expr(), |sum, &x| sum + x);
        sum * (1.0 / v.len() as f32)
    }
}
impl Transmittance for Vec3<f32> {
    fn transparent() -> Self {
        Vec3::splat(1.0)
    }
    fn opaque() -> Self {
        Vec3::splat(0.0)
    }
}
impl PartialTransmittance for Vec3<f32> {
    #[tracked]
    fn blend(a: Expr<Self>, b: Expr<Self>) -> Expr<Self> {
        (a + b) * 0.5
    }
    #[tracked]
    fn blend_n(v: &[Expr<Self>]) -> Expr<Self> {
        let sum = v
            .iter()
            .fold(Vec3::splat(0.0_f32).expr(), |sum, &x| sum + x);
        sum * (1.0 / v.len() as f32)
    }
}
impl Transmittance for Vec3<f16> {
    fn transparent() -> Self {
        Vec3::splat(f16::ONE)
    }
    fn opaque() -> Self {
        Vec3::splat(f16::ZERO)
    }
}
impl PartialTransmittance for Vec3<f16> {
    #[tracked]
    fn blend(a: Expr<Self>, b: Expr<Self>) -> Expr<Self> {
        (a + b) * f16::from_f32_const(0.5)
    }
    #[tracked]
    fn blend_n(v: &[Expr<Self>]) -> Expr<Self> {
        let sum = v
            .iter()
            .fold(Vec3::splat(f16::ZERO).expr(), |sum, &x| sum + x);
        sum * f16::from_f32(1.0 / v.len() as f32)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RgbF32T1;
impl FluenceType for RgbF32T1 {
    type Radiance = Vec3<f32>;
    type Transmittance = f32;
    #[tracked]
    fn over(near: Expr<Fluence<Self>>, far: Expr<Fluence<Self>>) -> Expr<Fluence<Self>> {
        Fluence::expr(
            near.transmittance * far.radiance + near.radiance,
            near.transmittance * far.transmittance,
        )
    }
    #[tracked]
    fn over_radiance(near: Expr<Fluence<Self>>, far: Expr<Self::Radiance>) -> Expr<Self::Radiance> {
        near.transmittance * far + near.radiance
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RgbF16;
impl FluenceType for RgbF16 {
    type Radiance = Vec3<f16>;
    type Transmittance = Vec3<f16>;
    #[tracked]
    fn over(near: Expr<Fluence<Self>>, far: Expr<Fluence<Self>>) -> Expr<Fluence<Self>> {
        Fluence::expr(
            near.transmittance * far.radiance + near.radiance,
            near.transmittance * far.transmittance,
        )
    }
    #[tracked]
    fn over_radiance(near: Expr<Fluence<Self>>, far: Expr<Self::Radiance>) -> Expr<Self::Radiance> {
        near.transmittance * far + near.radiance
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RgbF32;
impl FluenceType for RgbF32 {
    type Radiance = Vec3<f32>;
    type Transmittance = Vec3<f32>;
    #[tracked]
    fn over(near: Expr<Fluence<Self>>, far: Expr<Fluence<Self>>) -> Expr<Fluence<Self>> {
        Fluence::expr(
            near.transmittance * far.radiance + near.radiance,
            near.transmittance * far.transmittance,
        )
    }
    #[tracked]
    fn over_radiance(near: Expr<Fluence<Self>>, far: Expr<Self::Radiance>) -> Expr<Self::Radiance> {
        near.transmittance * far + near.radiance
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SingleF32;
impl FluenceType for SingleF32 {
    type Radiance = f32;
    type Transmittance = f32;
    #[tracked]
    fn over(near: Expr<Fluence<Self>>, far: Expr<Fluence<Self>>) -> Expr<Fluence<Self>> {
        Fluence::expr(
            near.transmittance * far.radiance + near.radiance,
            near.transmittance * far.transmittance,
        )
    }
    #[tracked]
    fn over_radiance(near: Expr<Fluence<Self>>, far: Expr<Self::Radiance>) -> Expr<Self::Radiance> {
        near.transmittance * far + near.radiance
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BinaryF32;
impl FluenceType for BinaryF32 {
    type Radiance = f32;
    type Transmittance = bool;
    #[tracked]
    fn over(near: Expr<Fluence<Self>>, far: Expr<Fluence<Self>>) -> Expr<Fluence<Self>> {
        Fluence::expr(
            near.transmittance.select(far.radiance, f32::black().expr()) + near.radiance,
            near.transmittance && far.transmittance,
        )
    }
    #[tracked]
    fn over_radiance(near: Expr<Fluence<Self>>, far: Expr<Self::Radiance>) -> Expr<Self::Radiance> {
        near.transmittance.select(far, near.radiance)
    }
}

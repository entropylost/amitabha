use keter::lang::types::vector::Vec3;
use keter::prelude::*;

use crate::fluence::{self, Fluence, FluenceType};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Value)]
pub struct Color<C: ColorType> {
    pub emission: C::Emission,
    pub opacity: C::Opacity,
}
impl<C: ColorType> Color<C> {
    pub fn empty() -> Self {
        Color {
            emission: C::Emission::black(),
            opacity: C::Opacity::transparent(),
        }
    }
    pub fn solid(emission: C::Emission) -> Self {
        Color {
            emission,
            opacity: C::Opacity::opaque(),
        }
    }
}
impl<C: ColorType> ColorExpr<C> {
    pub fn is_transparent(self) -> Expr<bool> {
        C::Opacity::is_transparent(self.opacity)
    }
}
impl<C: ColorType> ColorExpr<C> {
    pub fn to_fluence<F: FluenceType>(self, segment_length: Expr<f32>) -> Expr<Fluence<F>>
    where
        C: ToFluence<F>,
    {
        C::to_fluence(self.self_, segment_length)
    }
}

pub trait ColorType: 'static + Copy {
    type Emission: Emission;
    type Opacity: Opacity;
}

pub trait ToFluence<F: FluenceType>: ColorType {
    fn to_fluence(color: Expr<Color<Self>>, segment_length: Expr<f32>) -> Expr<Fluence<F>>;
}

pub trait Emission: Value {
    fn black() -> Self;
}
pub trait Opacity: Value {
    fn is_transparent(this: Expr<Self>) -> Expr<bool>;
    fn transparent() -> Self;
    fn opaque() -> Self;
}

impl Emission for f32 {
    fn black() -> Self {
        0.0
    }
}
impl Emission for Vec3<f32> {
    fn black() -> Self {
        Vec3::splat(0.0)
    }
}
impl Emission for Vec3<f16> {
    fn black() -> Self {
        Vec3::splat(f16::ZERO)
    }
}

impl Opacity for f32 {
    #[tracked]
    fn is_transparent(this: Expr<Self>) -> Expr<bool> {
        this == Self::transparent()
    }
    fn transparent() -> Self {
        0.0
    }
    fn opaque() -> Self {
        f32::INFINITY
    }
}
impl Opacity for Vec3<f32> {
    #[tracked]
    fn is_transparent(this: Expr<Self>) -> Expr<bool> {
        (this == Self::transparent()).all()
    }
    fn transparent() -> Self {
        Vec3::splat(0.0)
    }
    fn opaque() -> Self {
        Vec3::splat(f32::INFINITY)
    }
}
impl Opacity for Vec3<f16> {
    #[tracked]
    fn is_transparent(this: Expr<Self>) -> Expr<bool> {
        (this == Self::transparent()).all()
    }
    fn transparent() -> Self {
        Vec3::splat(f16::ZERO)
    }
    fn opaque() -> Self {
        Vec3::splat(f16::INFINITY)
    }
}
impl Opacity for bool {
    #[tracked]
    fn is_transparent(this: Expr<Self>) -> Expr<bool> {
        this == Self::transparent()
    }
    fn transparent() -> Self {
        false
    }
    fn opaque() -> Self {
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RgbF32;
impl ColorType for RgbF32 {
    type Emission = Vec3<f32>;
    type Opacity = Vec3<f32>;
}
impl ToFluence<fluence::RgbF32> for RgbF32 {
    #[tracked]
    fn to_fluence(
        color: Expr<Color<Self>>,
        segment_length: Expr<f32>,
    ) -> Expr<Fluence<fluence::RgbF32>> {
        let transmittance = (-color.opacity * segment_length).exp();
        Fluence::expr(color.emission * (1.0 - transmittance), transmittance)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RgbF16;
impl ColorType for RgbF16 {
    type Emission = Vec3<f16>;
    type Opacity = Vec3<f16>;
}
impl ToFluence<fluence::RgbF16> for RgbF16 {
    #[tracked]
    fn to_fluence(
        color: Expr<Color<Self>>,
        segment_length: Expr<f32>,
    ) -> Expr<Fluence<fluence::RgbF16>> {
        let transmittance = (-color.opacity * segment_length.cast_f16()).exp();
        Fluence::expr(color.emission * (f16::ONE - transmittance), transmittance)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SingleF32;
impl ColorType for SingleF32 {
    type Emission = f32;
    type Opacity = f32;
}
impl ToFluence<fluence::SingleF32> for SingleF32 {
    #[tracked]
    fn to_fluence(
        color: Expr<Color<Self>>,
        segment_length: Expr<f32>,
    ) -> Expr<Fluence<fluence::SingleF32>> {
        let transmittance = (-color.opacity * segment_length).exp();
        Fluence::expr(color.emission * (1.0 - transmittance), transmittance)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BinaryF32;
impl ColorType for BinaryF32 {
    type Emission = f32;
    type Opacity = bool;
}
impl ToFluence<fluence::BinaryF32> for BinaryF32 {
    #[tracked]
    fn to_fluence(
        color: Expr<Color<Self>>,
        segment_length: Expr<f32>,
    ) -> Expr<Fluence<fluence::BinaryF32>> {
        if segment_length < 0.01 {
            Fluence::empty().expr()
        } else {
            Fluence::expr(
                color.emission * color.opacity.cast_u32().cast_f32(),
                !color.opacity,
            )
        }
    }
}
impl ToFluence<fluence::SingleF32> for BinaryF32 {
    #[tracked]
    fn to_fluence(
        color: Expr<Color<Self>>,
        segment_length: Expr<f32>,
    ) -> Expr<Fluence<fluence::SingleF32>> {
        if segment_length < 0.01 {
            Fluence::empty().expr()
        } else {
            Fluence::expr(
                color.emission * color.opacity.cast_u32().cast_f32(),
                (!color.opacity).cast_u32().cast_f32(),
            )
        }
    }
}

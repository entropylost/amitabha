use keter::lang::types::vector::Vec3;
use keter::prelude::*;

use crate::fluence::{self, Fluence, FluenceType};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Value)]
pub struct Color<C: ColorType> {
    pub emission: C::Emission,
    pub opacity: C::Opacity,
}
impl<C: ColorType> ColorExpr<C> {
    pub fn is_transparent(self) -> Expr<bool> {
        C::Opacity::is_transparent(self.opacity)
    }
    pub fn to_fluence(self, segment_length: Expr<f32>) -> Expr<Fluence<C::Fluence>> {
        C::to_fluence(self.self_, segment_length)
    }
}

pub trait ColorType: 'static + Copy {
    type Emission: Emission;
    type Opacity: Opacity;
    type Fluence: FluenceType;

    fn to_fluence(
        color: Expr<Color<Self>>,
        segment_length: Expr<f32>,
    ) -> Expr<Fluence<Self::Fluence>>;
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
    type Fluence = fluence::RgbF32;
    #[tracked]
    fn to_fluence(
        color: Expr<Color<Self>>,
        segment_length: Expr<f32>,
    ) -> Expr<Fluence<Self::Fluence>> {
        let transmittance = (-color.opacity * segment_length).exp();
        Fluence::expr(color.emission * (1.0 - transmittance), transmittance)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RgbF16;
impl ColorType for RgbF16 {
    type Emission = Vec3<f16>;
    type Opacity = Vec3<f16>;
    type Fluence = fluence::RgbF16;
    #[tracked]
    fn to_fluence(
        color: Expr<Color<Self>>,
        segment_length: Expr<f32>,
    ) -> Expr<Fluence<Self::Fluence>> {
        let transmittance = (-color.opacity * segment_length.cast_f16()).exp();
        Fluence::expr(color.emission * (f16::ONE - transmittance), transmittance)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SingleF32;
impl ColorType for SingleF32 {
    type Emission = f32;
    type Opacity = f32;
    type Fluence = fluence::SingleF32;
    #[tracked]
    fn to_fluence(
        color: Expr<Color<Self>>,
        segment_length: Expr<f32>,
    ) -> Expr<Fluence<Self::Fluence>> {
        let transmittance = (-color.opacity * segment_length).exp();
        Fluence::expr(color.emission * (1.0 - transmittance), transmittance)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BinaryF32;
impl ColorType for BinaryF32 {
    type Emission = f32;
    type Opacity = bool;
    type Fluence = fluence::BinaryF32;
    #[tracked]
    fn to_fluence(
        color: Expr<Color<Self>>,
        _segment_length: Expr<f32>,
    ) -> Expr<Fluence<Self::Fluence>> {
        Fluence::expr(
            color.emission * color.opacity.cast_u32().cast_f32(),
            !color.opacity,
        )
    }
}

// TODO: Make ColorType<F> instead and implement on the preexisting fluence types?
#[derive(Debug, Clone, Copy)]
pub struct BinarySF32;
impl ColorType for BinarySF32 {
    type Emission = f32;
    type Opacity = bool;
    type Fluence = fluence::SingleF32;
    #[tracked]
    fn to_fluence(
        color: Expr<Color<Self>>,
        _segment_length: Expr<f32>,
    ) -> Expr<Fluence<Self::Fluence>> {
        Fluence::expr(
            color.emission * color.opacity.cast_u32().cast_f32(),
            (!color.opacity).cast_u32().cast_f32(),
        )
    }
}

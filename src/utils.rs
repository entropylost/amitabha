use super::*;

// https://github.com/markjarzynski/PCG3D/blob/master/pcg3d.hlsl
#[tracked]
pub fn pcg3d(v: Expr<Vec3<u32>>) -> Expr<Vec3<u32>> {
    let v = v.var();
    *v = v * 1664525u32 + 1013904223u32;

    *v.x += v.y * v.z;
    *v.y += v.z * v.x;
    *v.z += v.x * v.y;

    *v ^= v >> 16u32;

    *v.x += v.y * v.z;
    *v.y += v.z * v.x;
    *v.z += v.x * v.y;

    **v
}

#[tracked]
pub fn pcg(v: Expr<u32>) -> Expr<u32> {
    let state = v * 747796405u32 + 2891336453u32;
    let word = ((state >> ((state >> 28u32) + 4u32)) ^ state) * 277803737u32;
    (word >> 22u32) ^ word
}

#[tracked]
pub fn pcgf(v: Expr<u32>) -> Expr<f32> {
    pcg(v).cast_f32() / u32::MAX as f32
}

pub fn pcg_host(v: u32) -> u32 {
    let state = v.wrapping_mul(747796405u32).wrapping_add(2891336453u32);
    let word = ((state >> (state >> 28u32).wrapping_add(4u32)) ^ state).wrapping_mul(277803737u32);
    (word >> 22u32) ^ word
}

/*
Taken from: https://www.shadertoy.com/view/tlcSzs

vec3 LinearToSRGB ( vec3 col )
{
    return mix( col*12.92, 1.055*pow(col,vec3(1./2.4))-.055, step(.0031308,col) );
}

vec3 SRGBToLinear ( vec3 col )
{
    return mix( col/12.92, pow((col+.055)/1.055,vec3(2.4)), step(.04045,col) );
}
*/

#[tracked]
pub fn pcg3df(v: Expr<Vec3<u32>>) -> Expr<Vec3<f32>> {
    pcg3d(v).cast_f32() / u32::MAX as f32
}

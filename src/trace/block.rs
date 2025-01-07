use keter::lang::types::vector::Vec2;
use keter::prelude::*;

pub trait Block: Value {
    type Storage: IoTexel;
    const STORAGE_FORMAT: PixelStorage;
    const SIZE: u32;

    fn read(storage: &Tex2dView<Self::Storage>, offset: Expr<Vec2<u32>>) -> Expr<Self>;
    fn write(storage: &Tex2dView<Self::Storage>, offset: Expr<Vec2<u32>>, value: Expr<Self>);
    fn get(this: Expr<Self>, offset: Expr<Vec2<u32>>) -> Expr<bool>;
    fn set(this: Var<Self>, offset: Expr<Vec2<u32>>);
    fn is_empty(this: Expr<Self>) -> Expr<bool>;
    fn empty() -> Self;
}
impl Block for bool {
    type Storage = bool;
    const STORAGE_FORMAT: PixelStorage = PixelStorage::Byte1;
    const SIZE: u32 = 1;

    fn read(storage: &Tex2dView<Self::Storage>, offset: Expr<Vec2<u32>>) -> Expr<Self> {
        storage.read(offset)
    }
    fn write(storage: &Tex2dView<Self::Storage>, offset: Expr<Vec2<u32>>, value: Expr<Self>) {
        storage.write(offset, value);
    }
    fn get(this: Expr<Self>, _offset: Expr<Vec2<u32>>) -> Expr<bool> {
        this
    }
    #[tracked]
    fn set(this: Var<Self>, _offset: Expr<Vec2<u32>>) {
        *this = true;
    }
    #[tracked]
    fn is_empty(this: Expr<Self>) -> Expr<bool> {
        !this
    }
    fn empty() -> Self {
        false
    }
}
impl Block for u16 {
    type Storage = u32;
    const STORAGE_FORMAT: PixelStorage = PixelStorage::Short1;
    const SIZE: u32 = 4;

    fn read(storage: &Tex2dView<Self::Storage>, offset: Expr<Vec2<u32>>) -> Expr<Self> {
        storage.read(offset).cast_u16()
    }
    fn write(storage: &Tex2dView<Self::Storage>, offset: Expr<Vec2<u32>>, value: Expr<Self>) {
        storage.write(offset, value.cast_u32());
    }
    #[tracked]
    fn get(this: Expr<Self>, offset: Expr<Vec2<u32>>) -> Expr<bool> {
        this & (1 << (offset.x + offset.y * 4).cast_u16()) != 0
    }
    #[tracked]
    fn set(this: Var<Self>, offset: Expr<Vec2<u32>>) {
        *this |= 1 << (offset.x + offset.y * 4).cast_u16();
    }
    #[tracked]
    fn is_empty(this: Expr<Self>) -> Expr<bool> {
        this == 0
    }
    fn empty() -> Self {
        0
    }
}
impl Block for u64 {
    type Storage = Vec2<u32>;
    const STORAGE_FORMAT: PixelStorage = PixelStorage::Int2;
    const SIZE: u32 = 8;

    #[tracked]
    fn read(storage: &Tex2dView<Self::Storage>, offset: Expr<Vec2<u32>>) -> Expr<Self> {
        let v = storage.read(offset);
        v.x.cast_u64() | (v.y.cast_u64() << 32)
    }
    #[tracked]
    fn write(storage: &Tex2dView<Self::Storage>, offset: Expr<Vec2<u32>>, value: Expr<Self>) {
        storage.write(
            offset,
            Vec2::expr(value.cast_u32(), (value >> 32).cast_u32()),
        );
    }
    #[tracked]
    fn get(this: Expr<Self>, offset: Expr<Vec2<u32>>) -> Expr<bool> {
        this & (1 << (offset.x + offset.y * 8).cast_u64()) != 0
    }
    #[tracked]
    fn set(this: Var<Self>, offset: Expr<Vec2<u32>>) {
        *this |= 1 << (offset.x + offset.y * 8).cast_u64();
    }
    #[tracked]
    fn is_empty(this: Expr<Self>) -> Expr<bool> {
        this == 0
    }
    fn empty() -> Self {
        0
    }
}

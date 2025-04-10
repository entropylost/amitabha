use super::*;

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

type BlockType = u64;

pub struct VoxelTracer<C: ColorType>
where
    C::Emission: IoTexel,
    C::Opacity: IoTexel,
{
    emission: Tex2d<C::Emission>,
    opacity: Tex2d<C::Opacity>,
    pub diff: Tex2d<<BlockType as Block>::Storage>,
    pub diff_blocks: Tex2d<bool>,
    pub size: Vec2<u32>,
}
impl<C: ColorType> VoxelTracer<C>
where
    C::Emission: HasPixelStorage,
    C::Opacity: HasPixelStorage,
{
    pub fn new(size: Vec2<u32>) -> Self {
        Self::with_storage(
            size,
            C::Emission::natural_storage(),
            C::Opacity::natural_storage(),
        )
    }
}
impl<C: ColorType> VoxelTracer<C>
where
    C::Emission: IoTexel,
    C::Opacity: IoTexel,
{
    pub fn with_storage(
        size: Vec2<u32>,
        emission_storage: PixelStorage,
        opacity_storage: PixelStorage,
    ) -> Self {
        Self {
            emission: DEVICE.create_tex2d::<C::Emission>(emission_storage, size.x, size.y, 1),
            opacity: DEVICE.create_tex2d::<C::Opacity>(opacity_storage, size.x, size.y, 1),
            diff: DEVICE.create_tex2d::<<BlockType as Block>::Storage>(
                BlockType::STORAGE_FORMAT,
                size.x / BlockType::SIZE,
                size.y / BlockType::SIZE,
                1,
            ),
            diff_blocks: DEVICE.create_tex2d::<bool>(
                PixelStorage::Byte1,
                size.x / BlockType::SIZE,
                size.y / BlockType::SIZE,
                1,
            ),
            size,
        }
    }
    pub fn read(&self, pos: Expr<Vec2<u32>>) -> Expr<Color<C>> {
        Color::<C>::expr(self.emission.read(pos), self.opacity.read(pos))
    }
    pub fn read_emission(&self, pos: Expr<Vec2<u32>>) -> Expr<C::Emission> {
        self.emission.read(pos)
    }
    pub fn read_opacity(&self, pos: Expr<Vec2<u32>>) -> Expr<C::Opacity> {
        self.opacity.read(pos)
    }
    pub fn write(&self, pos: Expr<Vec2<u32>>, color: Expr<Color<C>>) {
        self.emission.write(pos, color.emission);
        self.opacity.write(pos, color.opacity);
    }
    pub fn write_emission(&self, pos: Expr<Vec2<u32>>, emission: Expr<C::Emission>) {
        self.emission.write(pos, emission);
    }
    pub fn write_opacity(&self, pos: Expr<Vec2<u32>>, opacity: Expr<C::Opacity>) {
        self.opacity.write(pos, opacity);
    }
}
const TRANSMITTANCE_CUTOFF: f16 = f16::from_f32_const(0.001);

impl VoxelTracer<crate::color::RgbF16> {
    #[tracked]
    pub fn compute_diff(&self) {
        let block = BlockType::empty().var();
        for dx in 0..BlockType::SIZE {
            for dy in 0..BlockType::SIZE {
                let pos = dispatch_id().xy() * BlockType::SIZE + Vec2::expr(dx, dy);
                let diff = false.var();
                let color = self.read(pos);
                for i in 0_u32..4_u32 {
                    let offset = [
                        Vec2::new(1, 0),
                        Vec2::new(-1, 0),
                        Vec2::new(0, 1),
                        Vec2::new(0, -1),
                    ]
                    .expr()[i];
                    let neighbor = pos.cast_i32() + offset;
                    if (neighbor >= 0).all() && (neighbor < self.size.expr().cast_i32()).all() {
                        let n_color = self.read(neighbor.cast_u32());
                        if (color.emission != n_color.emission).any()
                            || (color.opacity != n_color.opacity).any()
                        {
                            *diff = true;
                            break;
                        }
                    }
                }
                if diff {
                    BlockType::set(block, Vec2::expr(dx, dy));
                }
            }
        }
        self.diff_blocks
            .write(dispatch_id().xy(), !BlockType::is_empty(**block));
        BlockType::write(&self.diff.view(0), dispatch_id().xy(), **block);
    }
    #[tracked]
    pub fn trace_opt_to(
        &self,
        start: Expr<Vec2<f32>>,
        end: Expr<Vec2<f32>>,
    ) -> Expr<Fluence<crate::fluence::RgbF16>> {
        let delta = end - start;
        let length = delta.length();
        self.trace_opt(start, delta / length, length)
    }
    #[tracked]
    pub fn trace_opt(
        &self,
        start: Expr<Vec2<f32>>,
        ray_dir: Expr<Vec2<f32>>,
        length: Expr<f32>,
    ) -> Expr<Fluence<crate::fluence::RgbF16>> {
        let start = start + Vec2::new(0.001, 0.001);
        let inv_dir = (ray_dir + f32::EPSILON).recip();

        let interval = aabb_intersect(
            start,
            inv_dir,
            Vec2::splat(0.1).expr(),
            self.size.expr().cast_f32() - Vec2::splat(0.1).expr(),
        );
        let start_t = keter::max(interval.x, 0.0);
        let ray_start = start + start_t * ray_dir;
        let end_t = keter::min(interval.y, length) - start_t;
        if end_t <= 0.01 {
            Fluence::empty().expr()
        } else {
            let pos = ray_start.floor().cast_u32().var();

            let delta_dist = inv_dir.abs();
            let block_delta_dist = delta_dist * BlockType::SIZE as f32;

            let ray_step = ray_dir.signum().cast_i32().cast_u32();
            let side_dist =
                (ray_dir.signum() * (pos.cast_f32() - ray_start) + ray_dir.signum() * 0.5 + 0.5)
                    * delta_dist;
            let side_dist = side_dist.var();

            let block_offset = (ray_dir > 0.0).select(
                Vec2::splat_expr(0_u32),
                Vec2::splat_expr(BlockType::SIZE - 1),
            );

            let last_t = 0.0_f32.var();
            let fluence = Fluence::<crate::fluence::RgbF16>::empty().var();

            let finished = false.var();

            loop {
                loop {
                    let next_t = side_dist.reduce_min();

                    let block = BlockType::read(&self.diff.view(0), pos / BlockType::SIZE);

                    if BlockType::is_empty(block) {
                        break;
                    }

                    if BlockType::get(block, pos % BlockType::SIZE) || next_t >= end_t {
                        let segment_size = keter::min(next_t, end_t) - last_t;
                        let color = self.read(**pos);
                        *fluence = fluence.over(color.to_fluence(segment_size));

                        *last_t = next_t;

                        if (fluence.transmittance < TRANSMITTANCE_CUTOFF).all() {
                            *fluence.transmittance = Vec3::splat(f16::ZERO);
                            *finished = true;
                            break;
                        }

                        if next_t >= end_t {
                            *finished = true;
                            break;
                        }
                    }

                    let mask = side_dist <= side_dist.yx();

                    *side_dist += mask.select(delta_dist, Vec2::splat_expr(0.0));
                    *pos += mask.select(ray_step, Vec2::splat_expr(0));
                }

                if finished {
                    break;
                }

                let block_pos = (pos / BlockType::SIZE).var();
                let block_side_dist = (ray_dir.signum()
                    * (block_pos.cast_f32() - ray_start / BlockType::SIZE as f32)
                    + ray_dir.signum() * 0.5
                    + 0.5)
                    * block_delta_dist;
                let block_side_dist = block_side_dist.var();

                let next_t = block_side_dist.reduce_min().var();

                loop {
                    if next_t >= end_t {
                        let segment_size = end_t - last_t;
                        let color = self.read(**pos);
                        *fluence = fluence.over(color.to_fluence(segment_size));

                        *finished = true;
                        break;
                    }

                    let mask = block_side_dist <= block_side_dist.yx();

                    *block_side_dist += mask.select(block_delta_dist, Vec2::splat_expr(0.0));
                    *block_pos += mask.select(ray_step, Vec2::splat_expr(0));

                    let last_t = **next_t;
                    *next_t = block_side_dist.reduce_min();

                    if self.diff_blocks.read(block_pos) {
                        *pos = mask.select(
                            block_pos * BlockType::SIZE + block_offset,
                            (last_t * ray_dir + ray_start).floor().cast_u32(),
                        );
                        // let a = (pos / B::SIZE == block_pos).all();
                        // lc_assert!(a);
                        // This bugfix is necessary due to floating point issues.
                        if (pos / BlockType::SIZE != block_pos).any() {
                            // *fluence = Fluence::black();
                            *finished = true;
                        }
                        *side_dist = (ray_dir.signum() * (pos.cast_f32() - ray_start)
                            + ray_dir.signum() * 0.5
                            + 0.5)
                            * delta_dist;

                        break;
                    }
                }

                if finished {
                    break;
                }
            }
            **fluence
        }
    }
}
impl<C: ColorType, F: FluenceType> WorldTracer<F> for VoxelTracer<C>
where
    C: ToFluence<F>,
    C::Emission: IoTexel,
    C::Opacity: IoTexel,
{
    type Params = ();
    #[tracked]
    fn trace(
        &self,
        _params: &(),
        start: Expr<Vec2<f32>>,
        ray_dir: Expr<Vec2<f32>>,
        length: Expr<f32>,
    ) -> Expr<Fluence<F>> {
        // TODO: Remove, and instead bias the initialization.
        let start = start + Vec2::new(0.001, 0.001);
        let inv_dir = (ray_dir + f32::EPSILON).recip();
        let interval = aabb_intersect(
            start,
            inv_dir,
            Vec2::splat(0.1).expr(),
            self.size.expr().cast_f32() - Vec2::splat(0.1).expr(),
        );
        let start_t = keter::max(interval.x, 0.0);
        let ray_start = start + start_t * ray_dir;
        let end_t = keter::min(interval.y, length) - start_t;
        if end_t <= 0.01 {
            Fluence::<F>::empty().expr()
        } else {
            let pos = ray_start.floor().cast_u32().var();

            let delta_dist = inv_dir.abs();

            let ray_step = ray_dir.signum().cast_i32().cast_u32();
            let side_dist =
                (ray_dir.signum() * (pos.cast_f32() - ray_start) + ray_dir.signum() * 0.5 + 0.5)
                    * delta_dist;
            let side_dist = side_dist.var();

            let last_t = 0.0_f32.var();

            let fluence = Fluence::<F>::empty().var();

            loop {
                let next_t = side_dist.reduce_min();
                let color = self.read(**pos);
                let segment_size = keter::min(next_t, end_t) - last_t;
                *fluence = fluence.over(color.to_fluence(segment_size));
                *last_t = next_t;
                if next_t >= end_t {
                    break;
                }
                let mask = side_dist <= side_dist.yx();

                *side_dist += mask.select(delta_dist, Vec2::splat_expr(0.0));
                *pos += mask.select(ray_step, Vec2::splat_expr(0));
            }

            **fluence
        }
    }
}

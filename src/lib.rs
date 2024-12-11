#![allow(clippy::type_complexity)]

// TODO: Can change u32 to use to u16?

use std::marker::PhantomData;

use keter::graph::NodeConfigs;
use keter::lang::types::vector::Vec2;
use keter::prelude::*;
use keter::runtime::{AsKernelArg, KernelParameter};

pub mod fluence;
use fluence::{Fluence, FluenceType, Radiance};
pub mod color;
pub mod trace;
use trace::Tracer;
pub mod storage;
use storage::RadianceStorage;
pub mod utils;

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct Grid {
    pub size: Vec2<u32>,
    pub directions: u32,
}
impl Grid {
    pub fn new(size: Vec2<u32>, directions: u32) -> Grid {
        Grid { size, directions }
    }
    #[tracked]
    pub fn expr(size: Expr<Vec2<u32>>, directions: Expr<u32>) -> Expr<Grid> {
        Grid::from_comps_expr(GridComps { size, directions })
    }
}
impl GridExpr {
    #[tracked]
    pub fn next(self) -> Expr<Grid> {
        Grid::from_comps_expr(GridComps {
            size: Vec2::expr(self.size.x / 2, self.size.y),
            directions: self.directions * 2,
        })
    }
    #[tracked]
    pub fn last(self) -> Expr<Grid> {
        Grid::from_comps_expr(GridComps {
            size: Vec2::expr(self.size.x * 2, self.size.y),
            directions: self.directions / 2,
        })
    }

    #[tracked]
    pub fn ray_offset(self, dir: Expr<u32>) -> Expr<i32> {
        dir.cast_i32() - (self.directions / 2).cast_i32()
    }
    #[tracked]
    pub fn lower_offset(self, dir: Expr<u32>) -> Expr<i32> {
        dir.cast_i32() - (self.directions / 2).cast_i32()
    }
    #[tracked]
    pub fn upper_offset(self, dir: Expr<u32>) -> Expr<i32> {
        dir.cast_i32() - (self.directions / 2).cast_i32() + 1
    }
    #[tracked]
    pub fn offset(self, dir: Expr<f32>) -> Expr<f32> {
        dir - (self.directions / 2).cast_f32() + 0.5
    }
    #[tracked]
    pub fn ray_angle(self, spacing: Expr<f32>, dir: Expr<f32>) -> Expr<f32> {
        self.offset(dir).atan2(spacing)
    }
    #[tracked]
    pub fn angle_size(self, spacing: Expr<f32>, dir: Expr<u32>) -> Expr<f32> {
        let upper = self.ray_angle(spacing, dir.cast_f32() + 0.5);
        let lower = self.ray_angle(spacing, dir.cast_f32() - 0.5);
        upper - lower
    }
}

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct Probe {
    pub cell: Vec2<i32>,
    pub dir: u32,
}
impl Probe {
    pub fn expr(cell: Expr<Vec2<i32>>, dir: Expr<u32>) -> Expr<Probe> {
        Probe::from_comps_expr(ProbeComps { cell, dir })
    }
}
impl ProbeExpr {
    #[tracked]
    pub fn next(self) -> Expr<Probe> {
        Probe::expr(Vec2::expr(self.cell.x / 2, self.cell.y), self.dir)
    }
    #[tracked]
    pub fn with_dir(self, dir: Expr<u32>) -> Expr<Probe> {
        Probe::expr(self.cell, dir)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    CellX,
    CellY,
    Direction,
}
impl Axis {
    pub fn get(self, (cell, dir): (Expr<Vec2<u32>>, Expr<u32>)) -> Expr<u32> {
        match self {
            Axis::CellX => cell.x,
            Axis::CellY => cell.y,
            Axis::Direction => dir,
        }
    }
    #[tracked]
    pub fn join(
        mut axes: [Self; 3],
        probe: (Expr<Vec2<u32>>, Expr<u32>),
        size: (Expr<Vec2<u32>>, Expr<u32>),
    ) -> Expr<u32> {
        axes.reverse();
        let mut result = axes[0].get(probe);
        #[allow(unused_parens)]
        for &axis in &axes[1..] {
            result = result * axis.get(size) + axis.get(probe);
        }
        result
    }
    #[tracked]
    pub fn dispatch_id(axes: [Self; 3]) -> (Expr<Vec2<u32>>, Expr<u32>) {
        fn dispatch_id_n(n: usize) -> Expr<u32> {
            match n {
                0 => dispatch_id().x,
                1 => dispatch_id().y,
                2 => dispatch_id().z,
                _ => unreachable!(),
            }
        }
        let cell_x = dispatch_id_n(axes.iter().position(|&x| x == Axis::CellX).unwrap());
        let cell_y = dispatch_id_n(axes.iter().position(|&x| x == Axis::CellY).unwrap());
        let dir = dispatch_id_n(axes.iter().position(|&x| x == Axis::Direction).unwrap());
        (Vec2::expr(cell_x, cell_y), dir)
    }
    #[tracked]
    pub fn dispatch_size(axes: [Self; 3], grid: Grid) -> [u32; 3] {
        axes.map(|axis| match axis {
            Axis::CellX => grid.size.x,
            Axis::CellY => grid.size.y,
            Axis::Direction => grid.directions,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MergeKernelSettings<'a, F: FluenceType, T: Tracer<F>, S: RadianceStorage<F::Radiance>> {
    pub axes: [Axis; 3],
    pub block_size: [u32; 3],
    // TODO: Use boxing.
    pub storage: &'a S,
    pub tracer: &'a T,
    // pub split_even: bool,
    pub _marker: PhantomData<F>,
}

pub struct MergeKernel<F: FluenceType, T: Tracer<F>, S: RadianceStorage<F::Radiance>> {
    kernel: Kernel<
        fn(
            Grid,
            <T::Params as KernelParameter>::Arg,
            <S::Params as KernelParameter>::Arg,
            <S::Params as KernelParameter>::Arg,
        ),
    >,
    axes: [Axis; 3],
    _marker: PhantomData<F>,
}

impl<F: FluenceType, T: Tracer<F>, S: RadianceStorage<F::Radiance>>
    MergeKernelSettings<'_, F, T, S>
{
    pub fn build_kernel(&self) -> MergeKernel<F, T, S> {
        let kernel = DEVICE.create_kernel::<fn(
            Grid,
            <T::Params as KernelParameter>::Arg,
            <S::Params as KernelParameter>::Arg,
            <S::Params as KernelParameter>::Arg,
        )>(&track!(
            |grid, tracer_params, radiance_params, next_radiance_params| {
                set_block_size(self.block_size);
                let (cell, dir) = Axis::dispatch_id(self.axes);
                let cell = cell.cast_i32();
                let probe = Probe::expr(cell, dir);

                let value = merge(
                    grid,
                    probe,
                    (self.storage, &next_radiance_params),
                    (self.tracer, &tracer_params),
                );
                self.storage.store(&radiance_params, grid, probe, value);
            }
        ));
        MergeKernel {
            kernel,
            axes: self.axes,
            _marker: PhantomData,
        }
    }
}
impl<F: FluenceType, T: Tracer<F>, S: RadianceStorage<F::Radiance>> MergeKernel<F, T, S> {
    pub fn dispatch(
        &self,
        grid: Grid,
        tracer_params: &impl AsKernelArg<Output = <T::Params as KernelParameter>::Arg>,
        radiance_params: &impl AsKernelArg<Output = <S::Params as KernelParameter>::Arg>,
        next_radiance_params: &impl AsKernelArg<Output = <S::Params as KernelParameter>::Arg>,
    ) -> NodeConfigs<'static> {
        let dispatch_size = Axis::dispatch_size(self.axes, grid);
        self.kernel
            .dispatch_async(
                dispatch_size,
                &grid,
                tracer_params,
                radiance_params,
                next_radiance_params,
            )
            .into_node_configs()
    }
}

// TODO: Add tracer? Make separate Tracer1 type?
// Merges into (cell.x, cell.y + 0.5)
#[tracked]
pub fn merge_0_odd<F: FluenceType, S: RadianceStorage<F::Radiance>>(
    grid: Expr<Grid>,
    cell: Expr<Vec2<i32>>,
    (storage, next_radiance_params): (&S, &S::Params),
) -> Expr<F::Radiance> {
    let next_grid = grid.next();
    let load_next =
        |probe: Expr<Probe>| storage.load(next_radiance_params, next_grid, probe.next());

    let lower_dir = 0_u32.expr();
    let upper_dir = 1_u32.expr();
    let lower_offset = Vec2::expr(1, 0);
    let upper_offset = Vec2::expr(1, 1);

    let [lower, upper] = [Fluence::<F>::empty().expr(); 2];
    F::Radiance::merge(
        F::over_radiance(
            lower,
            load_next(Probe::expr(cell + lower_offset, lower_dir)),
        ),
        F::over_radiance(
            upper,
            load_next(Probe::expr(cell + upper_offset, upper_dir)),
        ),
    )
}

// Merges into (cell.x, cell.y)
#[tracked]
pub fn merge_0_even<F: FluenceType, S: RadianceStorage<F::Radiance>>(
    grid: Expr<Grid>,
    cell: Expr<Vec2<i32>>,
    (storage, next_radiance_params): (&S, &S::Params),
) -> Expr<F::Radiance> {
    let next_grid = grid.next();
    let load_next =
        |probe: Expr<Probe>| storage.load(next_radiance_params, next_grid, probe.next());

    let lower_dir = 0_u32.expr();
    let upper_dir = 1_u32.expr();

    let [lower, upper] = [Fluence::<F>::empty().expr(); 2];
    let next_cell = cell + Vec2::expr(2, 0);
    F::Radiance::merge(
        F::Radiance::blend(
            load_next(Probe::expr(cell, lower_dir)),
            F::over_radiance(
                lower,
                load_next(Probe::expr(next_cell - Vec2::expr(0, 1), lower_dir)),
            ),
        ),
        F::Radiance::blend(
            load_next(Probe::expr(cell, upper_dir)),
            F::over_radiance(
                upper,
                load_next(Probe::expr(next_cell + Vec2::expr(0, 1), upper_dir)),
            ),
        ),
    )
}

#[tracked]
pub fn merge<F: FluenceType, S: RadianceStorage<F::Radiance>, T: Tracer<F>>(
    grid: Expr<Grid>,
    probe: Expr<Probe>,
    (storage, next_radiance_params): (&S, &S::Params),
    (tracer, tracer_params): (&T, &T::Params),
) -> Expr<F::Radiance> {
    let next_grid = grid.next();
    let load_next =
        |probe: Expr<Probe>| storage.load(next_radiance_params, next_grid, probe.next());

    let cell = probe.cell;
    let dir = probe.dir;

    let lower_dir = dir * 2;
    let upper_dir = dir * 2 + 1;
    let lower_offset = Vec2::expr(1, grid.lower_offset(dir));
    let upper_offset = Vec2::expr(1, grid.upper_offset(dir));

    let factor = (cell.x % 2 == 0).select(2_i32.expr(), 1_i32.expr());

    let [lower, upper] = tracer.trace(tracer_params, grid, probe);

    let next_lower = F::over_radiance(
        lower,
        load_next(Probe::expr(cell + lower_offset * factor, lower_dir)),
    );
    let next_upper = F::over_radiance(
        upper,
        load_next(Probe::expr(cell + upper_offset * factor, upper_dir)),
    );

    if cell.x % 2 == 0 {
        // Could possibly be better as ((a + b) + (c + d)) * 0.5 instead of (a + b).mul_add(0.5, (c + d) * 0.5)
        // Or can reorder and compact with the next one.
        F::Radiance::merge(
            F::Radiance::blend(load_next(Probe::expr(cell, lower_dir)), next_lower),
            F::Radiance::blend(load_next(Probe::expr(cell, upper_dir)), next_upper),
        )
    } else {
        F::Radiance::merge(next_lower, next_upper)
    }
}

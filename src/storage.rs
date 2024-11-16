use keter::lang::types::vector::Vec2;
use keter::prelude::*;
use keter::runtime::KernelParameter;

use crate::color::Radiance;
use crate::{Grid, Probe};

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
        axes: [Self; 3],
        probe: (Expr<Vec2<u32>>, Expr<u32>),
        size: (Expr<Vec2<u32>>, Expr<u32>),
    ) -> Expr<u32> {
        let mut result = axes[0].get(probe);
        #[allow(unused_parens)]
        for &axis in &axes[1..] {
            result = result * axis.get(size) + axis.get(probe);
        }
        result
    }
}

// Note: Integer Multiplication is multiple instructions on compute capacity <= 6.2 (10-series and less).

// Construct from kernel arg somehow.
// Then later there's like a Pyraimd::construct_storages() and then Pyramid[level] passin.
pub trait RadianceStorage<R: Radiance> {
    type Params: KernelParameter;

    fn load(&self, params: &Self::Params, grid: Expr<Grid>, probe: Expr<Probe>) -> Expr<R>;
    fn store(&self, params: &Self::Params, grid: Expr<Grid>, probe: Expr<Probe>, radiance: Expr<R>);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferStorage {
    pub axes: [Axis; 3],
}

impl<R: Radiance> RadianceStorage<R> for BufferStorage {
    type Params = BufferVar<R>;

    #[tracked]
    fn load(&self, params: &Self::Params, grid: Expr<Grid>, probe: Expr<Probe>) -> Expr<R> {
        let cell = probe.cell.cast_u32();
        if (cell >= grid.size).any() {
            R::black().expr()
        } else {
            params.read(Axis::join(
                self.axes,
                (cell, probe.dir),
                (grid.size, grid.directions),
            ))
        }
    }
    #[tracked]
    fn store(
        &self,
        params: &Self::Params,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
        radiance: Expr<R>,
    ) {
        let cell = probe.cell.cast_u32();

        params.write(
            Axis::join(self.axes, (cell, probe.dir), (grid.size, grid.directions)),
            radiance,
        );
    }
}

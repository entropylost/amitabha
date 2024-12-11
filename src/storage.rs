use keter::prelude::*;
use keter::runtime::KernelParameter;

use crate::fluence::Radiance;
use crate::{Axis, Grid, Probe};

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

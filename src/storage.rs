use keter::prelude::*;
use keter::runtime::KernelParameter;

use crate::color::Radiance;
use crate::{Grid, Probe};

// Note: Integer Multiplication is multiple instructions on compute capacity <= 6.2 (10-series and less).

// Construct from kernel arg somehow.
// Then later there's like a Pyraimd::construct_storages() and then Pyramid[level] passin.
pub trait RadianceStorage<R: Radiance> {
    type Params: KernelParameter;

    fn load(&self, params: &Self::Params, grid: Expr<Grid>, probe: Expr<Probe>) -> Expr<R>;
    fn store(&self, params: &Self::Params, grid: Expr<Grid>, probe: Expr<Probe>, radiance: Expr<R>);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferStorage;

impl<R: Radiance> RadianceStorage<R> for BufferStorage {
    type Params = BufferVar<R>;

    #[tracked]
    fn load(&self, params: &Self::Params, grid: Expr<Grid>, probe: Expr<Probe>) -> Expr<R> {
        if (probe.cell >= grid.size).any() {
            R::black().expr()
        } else {
            params.read(
                probe.cell.x + probe.cell.y * grid.size.x + probe.dir * grid.size.x * grid.size.y,
            )
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
        params.write(
            probe.cell.x + probe.cell.y * grid.size.x + probe.dir * grid.size.x * grid.size.y,
            radiance,
        );
    }
}

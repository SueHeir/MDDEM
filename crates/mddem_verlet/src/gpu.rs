//! GPU velocity-Verlet integration. Built only with `--features gpu`.
//!
//! Mirrors the CPU integrators in [`crate::initial_integration`] and
//! [`crate::final_integration`]. Operates on the resident [`AtomGpu`]
//! buffers — no per-call host I/O. The sync systems registered by
//! `mddem_core::AtomGpuPlugin` make sure those buffers are current before /
//! after each GPU phase.

use sim_app::prelude::*;
use sim_scheduler::prelude::*;

use mddem_core::{Atom, ParticleSimScheduleSet};
use mddem_gpu::{AtomGpu, GpuRuntime};

use crate::{final_integration, initial_integration};

/// GPU replacement for [`initial_integration`].
pub fn initial_integration_gpu(
    rt: Res<GpuRuntime>,
    atoms: Res<Atom>,
    gpu: Res<AtomGpu>,
) {
    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 || !gpu.force.is_initialized() {
        return;
    }
    mddem_gpu::integrate_initial_resident(
        &rt,
        atoms.dt,
        gpu.inv_mass.handle(),
        gpu.force.handle(),
        gpu.vel.handle(),
        gpu.pos.handle(),
        nlocal,
    );
}

/// GPU replacement for [`final_integration`].
pub fn final_integration_gpu(
    rt: Res<GpuRuntime>,
    atoms: Res<Atom>,
    gpu: Res<AtomGpu>,
) {
    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 || !gpu.force.is_initialized() {
        return;
    }
    mddem_gpu::integrate_final_resident(
        &rt,
        atoms.dt,
        gpu.inv_mass.handle(),
        gpu.force.handle(),
        gpu.vel.handle(),
        nlocal,
    );
}

/// Plugin that swaps the CPU velocity-Verlet systems for GPU equivalents.
pub struct VelocityVerletGpuPlugin;

impl Plugin for VelocityVerletGpuPlugin {
    fn build(&self, app: &mut App) {
        app.remove_update_system(initial_integration)
            .remove_update_system(final_integration)
            .add_update_system(
                initial_integration_gpu,
                ParticleSimScheduleSet::InitialIntegration,
            )
            .add_update_system(
                final_integration_gpu,
                ParticleSimScheduleSet::FinalIntegration,
            );
    }
}

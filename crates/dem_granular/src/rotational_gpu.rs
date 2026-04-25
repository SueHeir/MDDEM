//! GPU quaternion-Verlet rotation. Built only with `--features gpu`.
//!
//! Operates on the resident [`DemAtomGpu`] buffers — no per-call host I/O.
//! Particles with `inv_inertia == 0.0` (clump sub-spheres) are skipped on
//! the GPU just as on the CPU.

use sim_app::prelude::*;
use sim_scheduler::prelude::*;

use mddem_core::{Atom, ParticleSimScheduleSet};
use mddem_gpu::{DemAtomGpu, GpuRuntime};

use super::rotational::{final_rotation, initial_rotation};

/// GPU replacement for [`initial_rotation`].
pub fn initial_rotation_gpu(
    rt: Res<GpuRuntime>,
    atoms: Res<Atom>,
    gpu: Res<DemAtomGpu>,
) {
    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 || !gpu.torque.is_initialized() {
        return;
    }
    mddem_gpu::rotate_initial_resident(
        &rt,
        atoms.dt,
        gpu.inv_inertia.handle(),
        gpu.torque.handle(),
        gpu.omega.handle(),
        gpu.quaternion.handle(),
        nlocal,
    );
}

/// GPU replacement for [`final_rotation`].
pub fn final_rotation_gpu(
    rt: Res<GpuRuntime>,
    atoms: Res<Atom>,
    gpu: Res<DemAtomGpu>,
) {
    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 || !gpu.torque.is_initialized() {
        return;
    }
    mddem_gpu::rotate_final_resident(
        &rt,
        atoms.dt,
        gpu.inv_inertia.handle(),
        gpu.torque.handle(),
        gpu.omega.handle(),
        nlocal,
    );
}

/// Plugin that swaps the CPU rotational systems for GPU equivalents.
pub struct RotationalDynamicsGpuPlugin;

impl Plugin for RotationalDynamicsGpuPlugin {
    fn build(&self, app: &mut App) {
        app.remove_update_system(initial_rotation)
            .remove_update_system(final_rotation)
            .add_update_system(
                initial_rotation_gpu,
                ParticleSimScheduleSet::InitialIntegration,
            )
            .add_update_system(
                final_rotation_gpu,
                ParticleSimScheduleSet::FinalIntegration,
            );
    }
}

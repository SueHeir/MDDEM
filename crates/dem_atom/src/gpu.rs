//! GPU plumbing for [`DemAtom`]. Built only with `--features gpu`.
//!
//! Provides [`DemAtomGpuPlugin`] — inserts the [`mddem_gpu::DemAtomGpu`]
//! resource and the sync systems that bridge `DemAtom` (in
//! `AtomDataRegistry`) ↔ GPU buffers at known schedule boundaries.

use sim_app::prelude::*;
use sim_scheduler::prelude::*;

use mddem_core::{Atom, AtomDataRegistry, ParticleSimScheduleSet};
use mddem_gpu::{DemAtomGpu, GpuRuntime};

use crate::DemAtom;

/// Upload `DemAtom.{torque, omega, quaternion, inv_inertia}` to GPU buffers.
/// `inv_inertia` is uploaded only on first call or after `nlocal` changes
/// (it's a per-atom constant set at insertion time).
pub fn sync_dem_atom_to_gpu(
    rt: Res<GpuRuntime>,
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    mut gpu: ResMut<DemAtomGpu>,
) {
    let dem = registry.expect::<DemAtom>("sync_dem_atom_to_gpu");
    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 {
        return;
    }
    gpu.torque.upload_vec3(&dem.torque[..nlocal], &rt);
    gpu.omega.upload_vec3(&dem.omega[..nlocal], &rt);
    gpu.quaternion.upload_vec4(&dem.quaternion[..nlocal], &rt);
    if !gpu.inv_inertia.is_initialized() || gpu.inv_inertia.nlocal() != nlocal {
        gpu.inv_inertia.upload_scalar(&dem.inv_inertia[..nlocal], &rt);
    }
    if !gpu.radius.is_initialized() || gpu.radius.nlocal() != nlocal {
        gpu.radius.upload_scalar(&dem.radius[..nlocal], &rt);
    }
}

/// Download `DemAtom.{torque, omega, quaternion}` from GPU back to CPU.
/// `inv_inertia` is read-only on the GPU side; no need to download. Batched
/// as a single `client.read(...)` call.
pub fn sync_dem_atom_to_cpu(
    rt: Res<GpuRuntime>,
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    gpu: Res<DemAtomGpu>,
) {
    let mut dem = registry.expect_mut::<DemAtom>("sync_dem_atom_to_cpu");
    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 || !gpu.torque.is_initialized() {
        return;
    }
    let torque_ptr = &mut dem.torque[..nlocal] as *mut [[f64; 3]];
    let omega_ptr = &mut dem.omega[..nlocal] as *mut [[f64; 3]];
    let quat_ptr = &mut dem.quaternion[..nlocal] as *mut [[f64; 4]];
    unsafe {
        gpu.download_torque_omega_quat(&rt, &mut *torque_ptr, &mut *omega_ptr, &mut *quat_ptr);
    }
}

/// Inserts the [`DemAtomGpu`] resource and registers the sync systems.
///
/// Sync points:
/// - `PreInitialIntegration` — upload (CPU is authoritative at start of step)
/// - `PreExchange` — download (CPU comm/force needs current state)
/// - `PreFinalIntegration` — upload (CPU just wrote new torques)
/// - `PostFinalIntegration` — download (next step's setup may read CPU)
pub struct DemAtomGpuPlugin;

impl Plugin for DemAtomGpuPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(DemAtomGpu::default())
            .add_update_system(
                sync_dem_atom_to_gpu,
                ParticleSimScheduleSet::PreInitialIntegration,
            )
            .add_update_system(sync_dem_atom_to_cpu, ParticleSimScheduleSet::PreExchange)
            .add_update_system(
                sync_dem_atom_to_gpu,
                ParticleSimScheduleSet::PreFinalIntegration,
            )
            .add_update_system(
                sync_dem_atom_to_cpu,
                ParticleSimScheduleSet::PostFinalIntegration,
            );
    }
}

//! GPU plumbing for `Atom`. Built only with `--features gpu`.
//!
//! Provides:
//! - [`AtomGpuPlugin`] — inserts the [`AtomGpu`] resource and the sync
//!   systems that bridge `Atom` ↔ GPU buffers at known schedule boundaries.
//! - [`ZeroForcesGpuPlugin`] — replaces the CPU [`zero_all_forces`] with a
//!   GPU kernel that operates on the resident `AtomGpu.force` buffer.
//!
//! Atom extension data (e.g. `DemAtom::torque`) is still zeroed CPU-side via
//! `AtomDataRegistry::zero_all` — the GPU path only owns the base `Atom`
//! force buffer here. Extension data is the responsibility of its own crate
//! (e.g. `dem_atom::DemAtomGpuPlugin`).

use sim_app::prelude::*;
use sim_scheduler::prelude::*;

use mddem_gpu::{AtomGpu, GpuField, GpuRuntime};

use crate::atom::{zero_all_forces, Atom, AtomDataRegistry, AtomPlugin};
use crate::schedule::ParticleSimScheduleSet;

/// GPU replacement for [`zero_all_forces`]. Reads `Res<AtomGpu>` for the
/// resident force buffer; no host I/O.
pub fn zero_all_forces_gpu(
    rt: Res<GpuRuntime>,
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    gpu: Res<AtomGpu>,
) {
    let n = atoms.len();
    if n > 0 && gpu.force.is_initialized() {
        // 3 scalar elements per particle.
        mddem_gpu::zero_resident(&rt, gpu.force.handle(), gpu.force.nlocal() * 3);
    }
    registry.zero_all(n);
}

/// Upload `Atom.{force, vel, pos, inv_mass}` to GPU buffers. `inv_mass` is
/// uploaded only on first call or when `nlocal` changes.
pub fn sync_atom_to_gpu(
    rt: Res<GpuRuntime>,
    atoms: Res<Atom>,
    mut gpu: ResMut<AtomGpu>,
) {
    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 {
        return;
    }
    gpu.force.upload_vec3(&atoms.force[..nlocal], &rt);
    gpu.vel.upload_vec3(&atoms.vel[..nlocal], &rt);
    gpu.pos.upload_vec3(&atoms.pos[..nlocal], &rt);
    if !gpu.inv_mass.is_initialized() || gpu.inv_mass.nlocal() != nlocal {
        gpu.inv_mass.upload_scalar(&atoms.inv_mass[..nlocal], &rt);
    }
    if !gpu.mass.is_initialized() || gpu.mass.nlocal() != nlocal {
        gpu.mass.upload_scalar(&atoms.mass[..nlocal], &rt);
    }
}

/// Download `Atom.{force, vel, pos}` from GPU back to CPU. Batched as a
/// single `client.read(...)` call to avoid N pipeline flushes on wgpu.
pub fn sync_atom_to_cpu(
    rt: Res<GpuRuntime>,
    mut atoms: ResMut<Atom>,
    gpu: Res<AtomGpu>,
) {
    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 || !gpu.force.is_initialized() {
        return;
    }
    // SAFETY: the three fields are distinct Vecs inside Atom, so a disjoint
    // borrow is safe.
    let force_ptr = &mut atoms.force[..nlocal] as *mut [[f64; 3]];
    let vel_ptr = &mut atoms.vel[..nlocal] as *mut [[f64; 3]];
    let pos_ptr = &mut atoms.pos[..nlocal] as *mut [[f64; 3]];
    unsafe {
        gpu.download_force_vel_pos(&rt, &mut *force_ptr, &mut *vel_ptr, &mut *pos_ptr);
    }
}

/// Inserts the [`AtomGpu`] resource and registers the sync systems at known
/// phase boundaries.
///
/// Sync points:
/// - `PreInitialIntegration` — upload (CPU is authoritative at start of step)
/// - `PreExchange` — download (CPU comm/neighbor/force needs current pos/vel)
/// - `PreFinalIntegration` — upload (CPU just wrote new force)
/// - `PostFinalIntegration` — download (next step's CPU work / output)
pub struct AtomGpuPlugin;

impl Plugin for AtomGpuPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(AtomGpu::default())
            .add_update_system(
                sync_atom_to_gpu,
                ParticleSimScheduleSet::PreInitialIntegration,
            )
            .add_update_system(sync_atom_to_cpu, ParticleSimScheduleSet::PreExchange)
            .add_update_system(
                sync_atom_to_gpu,
                ParticleSimScheduleSet::PreFinalIntegration,
            )
            .add_update_system(
                sync_atom_to_cpu,
                ParticleSimScheduleSet::PostFinalIntegration,
            );
    }
}

/// Plugin that swaps the CPU [`zero_all_forces`] for the GPU version.
///
/// Add **after** [`AtomPlugin`] (which registered the CPU system) and
/// **after** [`AtomGpuPlugin`] (which created the [`AtomGpu`] resource).
pub struct ZeroForcesGpuPlugin;

impl Plugin for ZeroForcesGpuPlugin {
    fn build(&self, app: &mut App) {
        app.remove_update_system(zero_all_forces)
            .add_update_system(zero_all_forces_gpu, ParticleSimScheduleSet::PostInitialIntegration);
    }
}

#[allow(dead_code)]
fn _docs_link(_p: &AtomPlugin, _f: &GpuField) {}

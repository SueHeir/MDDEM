//! Resident GPU buffers for `Atom` and atom extensions.
//!
//! [`GpuField`] wraps a single cubecl [`Handle`] plus the `nlocal` it was
//! last sized to. [`AtomGpu`] bundles the per-`Atom` fields needed by the
//! integration / force kernels.
//!
//! ```text
//!   PreInitialIntegration   → sync_atom_to_gpu (CPU → GPU, all fields)
//!   InitialIntegration      → integrate / rotate kernels read & write GPU
//!   PostInitialIntegration  → zero_forces kernel writes GPU
//!   PreExchange             → sync_atom_to_cpu (GPU → CPU, vel/pos for comm)
//!   Force (CPU)             → contact force writes Atom.force
//!   PreFinalIntegration     → sync_atom_to_gpu (force back to GPU)
//!   FinalIntegration        → kernels read & write GPU
//!   PostFinalIntegration    → sync_atom_to_cpu (vel/pos for output)
//! ```
//!
//! No internal dirty-state tracking — the `sync_*` systems are placed at
//! known schedule boundaries and do unconditional uploads/downloads of all
//! fields. Cheaper than per-kernel uploads (the v1 hybrid path), and simpler
//! than tracking dirty bits.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::{
    flatten_vec3_f32, flatten_vec3_f64, flatten_vec4_f32, flatten_vec4_f64, unflatten_vec3_f32,
    unflatten_vec3_f64, unflatten_vec4_f32, unflatten_vec4_f64, GpuRuntime,
};

/// One GPU buffer + last-uploaded `nlocal`. Allocated lazily on first upload.
#[derive(Default)]
pub struct GpuField {
    handle: Option<Handle>,
    nlocal: usize,
}

impl GpuField {
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the underlying handle. Panics if no upload has happened yet —
    /// systems should always run after a `sync_*_to_gpu` system.
    pub fn handle(&self) -> &Handle {
        self.handle
            .as_ref()
            .expect("GpuField has no handle yet — did the sync system run?")
    }

    pub fn nlocal(&self) -> usize {
        self.nlocal
    }

    /// True iff a buffer is allocated.
    pub fn is_initialized(&self) -> bool {
        self.handle.is_some()
    }

    // ── vec3 ────────────────────────────────────────────────────────────────

    pub fn upload_vec3(&mut self, data: &[[f64; 3]], rt: &GpuRuntime) {
        match rt {
            GpuRuntime::Wgpu(client) => {
                let flat = flatten_vec3_f32(data);
                self.handle = Some(client.create_from_slice(f32::as_bytes(&flat)));
            }
            GpuRuntime::Cpu(client) => {
                let flat = flatten_vec3_f64(data);
                self.handle = Some(client.create_from_slice(f64::as_bytes(&flat)));
            }
        }
        self.nlocal = data.len();
    }

    pub fn download_vec3(&self, out: &mut [[f64; 3]], rt: &GpuRuntime) {
        let h = self.handle().clone();
        match rt {
            GpuRuntime::Wgpu(client) => {
                let bytes = client.read_one_unchecked(h);
                unflatten_vec3_f32(f32::from_bytes(&bytes), out);
            }
            GpuRuntime::Cpu(client) => {
                let bytes = client.read_one_unchecked(h);
                unflatten_vec3_f64(f64::from_bytes(&bytes), out);
            }
        }
    }

    // ── vec4 ────────────────────────────────────────────────────────────────

    pub fn upload_vec4(&mut self, data: &[[f64; 4]], rt: &GpuRuntime) {
        match rt {
            GpuRuntime::Wgpu(client) => {
                let flat = flatten_vec4_f32(data);
                self.handle = Some(client.create_from_slice(f32::as_bytes(&flat)));
            }
            GpuRuntime::Cpu(client) => {
                let flat = flatten_vec4_f64(data);
                self.handle = Some(client.create_from_slice(f64::as_bytes(&flat)));
            }
        }
        self.nlocal = data.len();
    }

    pub fn download_vec4(&self, out: &mut [[f64; 4]], rt: &GpuRuntime) {
        let h = self.handle().clone();
        match rt {
            GpuRuntime::Wgpu(client) => {
                let bytes = client.read_one_unchecked(h);
                unflatten_vec4_f32(f32::from_bytes(&bytes), out);
            }
            GpuRuntime::Cpu(client) => {
                let bytes = client.read_one_unchecked(h);
                unflatten_vec4_f64(f64::from_bytes(&bytes), out);
            }
        }
    }

    // ── scalar f64 ──────────────────────────────────────────────────────────

    pub fn upload_scalar(&mut self, data: &[f64], rt: &GpuRuntime) {
        match rt {
            GpuRuntime::Wgpu(client) => {
                let flat: Vec<f32> = data.iter().map(|x| *x as f32).collect();
                self.handle = Some(client.create_from_slice(f32::as_bytes(&flat)));
            }
            GpuRuntime::Cpu(client) => {
                self.handle = Some(client.create_from_slice(f64::as_bytes(data)));
            }
        }
        self.nlocal = data.len();
    }

    #[allow(dead_code)]
    pub fn download_scalar(&self, out: &mut [f64], rt: &GpuRuntime) {
        let h = self.handle().clone();
        match rt {
            GpuRuntime::Wgpu(client) => {
                let bytes = client.read_one_unchecked(h);
                let f32_data = f32::from_bytes(&bytes);
                for (dst, src) in out.iter_mut().zip(f32_data) {
                    *dst = *src as f64;
                }
            }
            GpuRuntime::Cpu(client) => {
                let bytes = client.read_one_unchecked(h);
                out.copy_from_slice(f64::from_bytes(&bytes));
            }
        }
    }
}

/// Resident GPU mirror of the `Atom` fields needed by integration / force.
///
/// Insert as a resource via `mddem_core::AtomGpuPlugin` (gated by `gpu`
/// feature). `inv_mass` and `mass` are uploaded once at setup since they're
/// read-only.
#[derive(Default)]
pub struct AtomGpu {
    pub force: GpuField,
    pub vel: GpuField,
    pub pos: GpuField,
    pub inv_mass: GpuField,
    pub mass: GpuField,
}

impl AtomGpu {
    /// Single-flush batched download of `force`, `vel`, `pos`. On wgpu/Metal
    /// each `read_one` is a pipeline flush; batching N reads into one call
    /// saves N-1 flushes per sync system.
    pub fn download_force_vel_pos(
        &self,
        rt: &GpuRuntime,
        out_force: &mut [[f64; 3]],
        out_vel: &mut [[f64; 3]],
        out_pos: &mut [[f64; 3]],
    ) {
        let f = self.force.handle().clone();
        let v = self.vel.handle().clone();
        let p = self.pos.handle().clone();
        match rt {
            GpuRuntime::Wgpu(client) => {
                let bytes = client.read(vec![f, v, p]);
                unflatten_vec3_f32(f32::from_bytes(&bytes[0]), out_force);
                unflatten_vec3_f32(f32::from_bytes(&bytes[1]), out_vel);
                unflatten_vec3_f32(f32::from_bytes(&bytes[2]), out_pos);
            }
            GpuRuntime::Cpu(client) => {
                let bytes = client.read(vec![f, v, p]);
                unflatten_vec3_f64(f64::from_bytes(&bytes[0]), out_force);
                unflatten_vec3_f64(f64::from_bytes(&bytes[1]), out_vel);
                unflatten_vec3_f64(f64::from_bytes(&bytes[2]), out_pos);
            }
        }
    }
}

/// Resident GPU mirror of the `DemAtom` extension fields needed by rotation
/// and contact force.
///
/// Insert as a resource via `dem_atom::DemAtomGpuPlugin`. `inv_inertia` and
/// `radius` are uploaded once at setup since they're read-only.
#[derive(Default)]
pub struct DemAtomGpu {
    pub torque: GpuField,
    pub omega: GpuField,
    pub quaternion: GpuField,
    pub inv_inertia: GpuField,
    pub radius: GpuField,
}

impl DemAtomGpu {
    /// Single-flush batched download of `torque`, `omega`, `quaternion`.
    pub fn download_torque_omega_quat(
        &self,
        rt: &GpuRuntime,
        out_torque: &mut [[f64; 3]],
        out_omega: &mut [[f64; 3]],
        out_quat: &mut [[f64; 4]],
    ) {
        let t = self.torque.handle().clone();
        let o = self.omega.handle().clone();
        let q = self.quaternion.handle().clone();
        match rt {
            GpuRuntime::Wgpu(client) => {
                let bytes = client.read(vec![t, o, q]);
                unflatten_vec3_f32(f32::from_bytes(&bytes[0]), out_torque);
                unflatten_vec3_f32(f32::from_bytes(&bytes[1]), out_omega);
                unflatten_vec4_f32(f32::from_bytes(&bytes[2]), out_quat);
            }
            GpuRuntime::Cpu(client) => {
                let bytes = client.read(vec![t, o, q]);
                unflatten_vec3_f64(f64::from_bytes(&bytes[0]), out_torque);
                unflatten_vec3_f64(f64::from_bytes(&bytes[1]), out_omega);
                unflatten_vec4_f64(f64::from_bytes(&bytes[2]), out_quat);
            }
        }
    }
}

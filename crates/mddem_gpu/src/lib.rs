//! Generic GPU compute layer for MDDEM.
//!
//! MDDEM is an `f64` codebase, but most consumer GPUs (Metal, wgpu, integrated
//! Intel/AMD) either don't support `f64` at all or run it at 1/16–1/32 the
//! `f32` rate. This crate lets us write a single `#[cube]` kernel generic over
//! a [`Float`] bound and pick the precision *at runtime* based on what the
//! detected backend can actually do.
//!
//! - **CUDA / ROCm / CPU runtime** → `f64` (matches the rest of MDDEM bit-for-bit
//!   in the limit; production path).
//! - **wgpu (Metal / Vulkan / DX12 on consumer GPUs)** → `f32` fallback. Useful
//!   for development and validation on a laptop, **not** for long DEM contact
//!   integrations where drift compounds.
//!
//! ## Layout
//!
//! - [`GpuRuntime`] is the resource that holds the cubecl client. Insert it
//!   into the [`App`] via [`GpuRuntimePlugin`].
//! - Slice launchers ([`integrate_initial`], [`rotate_initial`], etc.) take
//!   `&GpuRuntime`, dispatch on backend, and operate on plain Rust slices.
//!   Domain crates wrap these in `Plugin`s under their own `gpu` feature.
//!
//! ## Numerical caveat
//!
//! Don't compare `f32`-backend results against the `f64` reference with the
//! same tolerance you'd use CPU↔CUDA. Plan for ~1e-6 relative drift per step
//! on the f32 path and gate any regression tests accordingly.

use cubecl::prelude::*;
use sim_app::prelude::*;

pub mod atom_gpu;
pub use atom_gpu::{AtomGpu, DemAtomGpu, GpuField};

// ── Runtime precision selection ─────────────────────────────────────────────

/// Which floating-point precision a launched kernel should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    F64,
}

/// Which cubecl backend was selected, for logging / dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// `cubecl-wgpu` — Metal / Vulkan / DX12. Consumer GPUs, f32 only.
    Wgpu,
    /// `cubecl-cuda` — NVIDIA, full f64.
    Cuda,
    /// `cubecl-hip` — AMD ROCm, full f64.
    Hip,
    /// `cubecl-cpu` — MLIR CPU runtime, f64. Local f64 reference on machines
    /// without a CUDA-capable GPU.
    Cpu,
}

impl BackendKind {
    /// Best precision this backend natively supports without emulation.
    pub fn native_precision(self) -> Precision {
        match self {
            BackendKind::Wgpu => Precision::F32,
            BackendKind::Cuda | BackendKind::Hip | BackendKind::Cpu => Precision::F64,
        }
    }
}

/// Pick precision by intersecting what the user asked for with what the
/// backend can actually run. A user requesting `F64` on a `Wgpu` backend gets
/// `F32` — kernels never silently launch at a precision they can't execute.
pub fn select_precision(backend: BackendKind, requested: Option<Precision>) -> Precision {
    let native = backend.native_precision();
    match (requested, native) {
        (Some(Precision::F64), Precision::F32) => Precision::F32,
        (Some(p), _) => p,
        (None, p) => p,
    }
}

// ── GpuRuntime resource ─────────────────────────────────────────────────────

/// Runtime-selected GPU backend, holding a cubecl `ComputeClient`.
///
/// Inserted into the [`App`] as a resource by [`GpuRuntimePlugin`]. Borrowed
/// by GPU systems via `Res<GpuRuntime>`. Construction is one-shot per app;
/// don't call `GpuRuntime::new(...)` per step (each call recreates the
/// underlying device context, which is expensive).
pub enum GpuRuntime {
    /// wgpu (Metal / Vulkan / DX12) — f32 only.
    Wgpu(ComputeClient<cubecl::wgpu::WgpuRuntime>),
    /// cubecl CPU (MLIR) — f64.
    Cpu(ComputeClient<cubecl::cpu::CpuRuntime>),
}

impl GpuRuntime {
    /// Construct a runtime for the requested backend. Panics if the backend
    /// isn't supported in this build (e.g. `Cuda` without the `cuda` feature).
    pub fn new(backend: BackendKind) -> Self {
        match backend {
            BackendKind::Wgpu => {
                let device = cubecl::wgpu::WgpuDevice::default();
                Self::Wgpu(cubecl::wgpu::WgpuRuntime::client(&device))
            }
            BackendKind::Cpu => {
                let device = cubecl::cpu::CpuDevice;
                Self::Cpu(cubecl::cpu::CpuRuntime::client(&device))
            }
            BackendKind::Cuda => panic!("Cuda backend requires `cuda` feature"),
            BackendKind::Hip => panic!("Hip backend requires `hip` feature"),
        }
    }

    pub fn backend(&self) -> BackendKind {
        match self {
            Self::Wgpu(_) => BackendKind::Wgpu,
            Self::Cpu(_) => BackendKind::Cpu,
        }
    }

    pub fn precision(&self) -> Precision {
        self.backend().native_precision()
    }
}

/// `Plugin` that constructs a [`GpuRuntime`] for the given backend and
/// inserts it as a resource. Add this BEFORE any GPU system plugins so
/// `Res<GpuRuntime>` is available when their `build()` runs.
pub struct GpuRuntimePlugin {
    pub backend: BackendKind,
}

impl GpuRuntimePlugin {
    pub fn new(backend: BackendKind) -> Self {
        Self { backend }
    }
}

impl Plugin for GpuRuntimePlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(GpuRuntime::new(self.backend));
    }
}

// ── Common kernel launch helpers ────────────────────────────────────────────

pub(crate) const BLOCK: u32 = 256;

pub(crate) fn cube_count_for(n: usize) -> CubeCount {
    let blocks = (n as u32).div_ceil(BLOCK);
    CubeCount::Static(blocks, 1, 1)
}

// ── vec3 / vec4 ↔ flat helpers ──────────────────────────────────────────────

pub(crate) fn flatten_vec3_f32(input: &[[f64; 3]]) -> Vec<f32> {
    input.iter().flat_map(|v| v.iter().map(|x| *x as f32)).collect()
}

pub(crate) fn flatten_vec3_f64(input: &[[f64; 3]]) -> Vec<f64> {
    input.iter().flat_map(|v| v.iter().copied()).collect()
}

pub(crate) fn unflatten_vec3_f32(flat: &[f32], output: &mut [[f64; 3]]) {
    for (i, dst) in output.iter_mut().enumerate() {
        let base = i * 3;
        *dst = [flat[base] as f64, flat[base + 1] as f64, flat[base + 2] as f64];
    }
}

pub(crate) fn unflatten_vec3_f64(flat: &[f64], output: &mut [[f64; 3]]) {
    for (i, dst) in output.iter_mut().enumerate() {
        let base = i * 3;
        *dst = [flat[base], flat[base + 1], flat[base + 2]];
    }
}

pub(crate) fn flatten_vec4_f32(input: &[[f64; 4]]) -> Vec<f32> {
    input.iter().flat_map(|v| v.iter().map(|x| *x as f32)).collect()
}

pub(crate) fn flatten_vec4_f64(input: &[[f64; 4]]) -> Vec<f64> {
    input.iter().flat_map(|v| v.iter().copied()).collect()
}

pub(crate) fn unflatten_vec4_f32(flat: &[f32], output: &mut [[f64; 4]]) {
    for (i, dst) in output.iter_mut().enumerate() {
        let base = i * 4;
        *dst = [
            flat[base] as f64,
            flat[base + 1] as f64,
            flat[base + 2] as f64,
            flat[base + 3] as f64,
        ];
    }
}

pub(crate) fn unflatten_vec4_f64(flat: &[f64], output: &mut [[f64; 4]]) {
    for (i, dst) in output.iter_mut().enumerate() {
        let base = i * 4;
        *dst = [flat[base], flat[base + 1], flat[base + 2], flat[base + 3]];
    }
}

// ── Kernels ─────────────────────────────────────────────────────────────────

/// Toy SAXPY: `y[i] = a * x[i] + y[i]`. Generic over precision; canary for
/// the cubecl plumbing.
#[cube(launch_unchecked)]
pub fn saxpy_kernel<F: Float + CubeElement>(a: F, x: &Array<F>, y: &mut Array<F>) {
    if ABSOLUTE_POS < x.len() {
        y[ABSOLUTE_POS] = a * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}

/// Writes `0.0` to every element of `buf`.
#[cube(launch_unchecked)]
pub fn zero_kernel<F: Float + CubeElement>(buf: &mut Array<F>) {
    if ABSOLUTE_POS < buf.len() {
        buf[ABSOLUTE_POS] = F::new(0.0);
    }
}

/// Velocity-Verlet first half: half-step velocity kick + full-step position
/// drift. One thread per particle; each handles 3 scalars (x, y, z).
#[cube(launch_unchecked)]
pub fn integrate_initial_kernel<F: Float + CubeElement>(
    dt: F,
    inv_mass: &Array<F>,
    force: &Array<F>,
    vel: &mut Array<F>,
    pos: &mut Array<F>,
) {
    let i = ABSOLUTE_POS;
    if i < inv_mass.len() {
        let half_dt_over_m = F::new(0.5) * dt * inv_mass[i];
        let base = i * 3;
        vel[base] += half_dt_over_m * force[base];
        vel[base + 1] += half_dt_over_m * force[base + 1];
        vel[base + 2] += half_dt_over_m * force[base + 2];
        pos[base] += vel[base] * dt;
        pos[base + 1] += vel[base + 1] * dt;
        pos[base + 2] += vel[base + 2] * dt;
    }
}

/// Velocity-Verlet second half: completing the velocity kick.
#[cube(launch_unchecked)]
pub fn integrate_final_kernel<F: Float + CubeElement>(
    dt: F,
    inv_mass: &Array<F>,
    force: &Array<F>,
    vel: &mut Array<F>,
) {
    let i = ABSOLUTE_POS;
    if i < inv_mass.len() {
        let half_dt_over_m = F::new(0.5) * dt * inv_mass[i];
        let base = i * 3;
        vel[base] += half_dt_over_m * force[base];
        vel[base + 1] += half_dt_over_m * force[base + 1];
        vel[base + 2] += half_dt_over_m * force[base + 2];
    }
}

/// Rotational Verlet first half: ω half-kick + quaternion update via
/// axis-angle. Particles with `inv_inertia == 0.0` are skipped (clump
/// sub-spheres). Quaternions are stored `[w, x, y, z]`.
#[cube(launch_unchecked)]
pub fn rotate_initial_kernel<F: Float + CubeElement>(
    dt: F,
    inv_inertia: &Array<F>,
    torque: &Array<F>,
    omega: &mut Array<F>,
    quaternion: &mut Array<F>,
) {
    let i = ABSOLUTE_POS;
    if i < inv_inertia.len() {
        let inv_i = inv_inertia[i];
        if inv_i != F::new(0.0) {
            let base3 = i * 3;
            let half_dt_inv_i = F::new(0.5) * dt * inv_i;

            omega[base3] += half_dt_inv_i * torque[base3];
            omega[base3 + 1] += half_dt_inv_i * torque[base3 + 1];
            omega[base3 + 2] += half_dt_inv_i * torque[base3 + 2];

            let ox = omega[base3];
            let oy = omega[base3 + 1];
            let oz = omega[base3 + 2];
            let omega_mag = F::sqrt(ox * ox + oy * oy + oz * oz);
            let angle = omega_mag * dt;

            if angle > F::new(1e-14) {
                let inv = F::new(1.0) / omega_mag;
                let ax = ox * inv;
                let ay = oy * inv;
                let az = oz * inv;
                let half = angle * F::new(0.5);
                let s = F::sin(half);
                let dqw = F::cos(half);
                let dqx = ax * s;
                let dqy = ay * s;
                let dqz = az * s;

                let base4 = i * 4;
                let qw = quaternion[base4];
                let qx = quaternion[base4 + 1];
                let qy = quaternion[base4 + 2];
                let qz = quaternion[base4 + 3];

                quaternion[base4] = dqw * qw - dqx * qx - dqy * qy - dqz * qz;
                quaternion[base4 + 1] = dqw * qx + dqx * qw + dqy * qz - dqz * qy;
                quaternion[base4 + 2] = dqw * qy - dqx * qz + dqy * qw + dqz * qx;
                quaternion[base4 + 3] = dqw * qz + dqx * qy - dqy * qx + dqz * qw;
            }
        }
    }
}

/// Rotational Verlet second half: ω half-kick using updated torques.
#[cube(launch_unchecked)]
pub fn rotate_final_kernel<F: Float + CubeElement>(
    dt: F,
    inv_inertia: &Array<F>,
    torque: &Array<F>,
    omega: &mut Array<F>,
) {
    let i = ABSOLUTE_POS;
    if i < inv_inertia.len() {
        let inv_i = inv_inertia[i];
        if inv_i != F::new(0.0) {
            let base3 = i * 3;
            let half_dt_inv_i = F::new(0.5) * dt * inv_i;
            omega[base3] += half_dt_inv_i * torque[base3];
            omega[base3 + 1] += half_dt_inv_i * torque[base3 + 1];
            omega[base3 + 2] += half_dt_inv_i * torque[base3 + 2];
        }
    }
}

/// Hertz normal contact force only (NO friction, NO Mindlin, NO contact
/// history). Single material — `e_eff` and `beta` are scalar parameters, not
/// per-pair tables. Each thread iterates its local atom's neighbor list and
/// accumulates pair forces into a register, then writes once at the end.
///
/// **Requires a full-pair neighbor list** (Newton-3 disabled) so each thread
/// only writes its own atom's force entry — no atomics needed. With a
/// half-pair list this kernel would silently lose half the force
/// contributions.
///
/// `n_atoms` is the number of atoms with a position entry (local + ghost).
/// `n_local` is the number that get a force write.
#[cube(launch_unchecked)]
pub fn hertz_normal_kernel<F: Float + CubeElement>(
    e_eff: F,
    beta: F,
    pos: &Array<F>,
    vel: &Array<F>,
    radius: &Array<F>,
    inv_mass: &Array<F>,
    neighbor_offsets: &Array<u32>,
    neighbor_indices: &Array<u32>,
    force: &mut Array<F>,
) {
    let i = ABSOLUTE_POS;
    if i < neighbor_offsets.len() - 1 {
        let i3 = i * 3;
        let xi = pos[i3];
        let yi = pos[i3 + 1];
        let zi = pos[i3 + 2];
        let vxi = vel[i3];
        let vyi = vel[i3 + 1];
        let vzi = vel[i3 + 2];
        let ri = radius[i];
        let inv_mi = inv_mass[i];
        let mi = if inv_mi > F::new(0.0) {
            F::new(1.0) / inv_mi
        } else {
            F::new(1.0e30)
        };

        let mut fxi = F::new(0.0);
        let mut fyi = F::new(0.0);
        let mut fzi = F::new(0.0);

        let start = neighbor_offsets[i];
        let end = neighbor_offsets[i + 1];

        for nidx in start..end {
            let j = neighbor_indices[nidx as usize] as usize;
            let j3 = j * 3;
            let dx = pos[j3] - xi;
            let dy = pos[j3 + 1] - yi;
            let dz = pos[j3 + 2] - zi;
            let r2 = dx * dx + dy * dy + dz * dz;
            let r_sum = ri + radius[j];

            if r2 < r_sum * r_sum && r2 > F::new(0.0) {
                let dist = F::sqrt(r2);
                let inv_dist = F::new(1.0) / dist;
                let nx = dx * inv_dist;
                let ny = dy * inv_dist;
                let nz = dz * inv_dist;
                let delta = r_sum - dist;

                let r_eff = (ri * radius[j]) / r_sum;
                let kn = (F::new(4.0) / F::new(3.0)) * e_eff * F::sqrt(r_eff * delta);
                let fn_spring = kn * delta;

                let inv_mj = inv_mass[j];
                let mj = if inv_mj > F::new(0.0) {
                    F::new(1.0) / inv_mj
                } else {
                    F::new(1.0e30)
                };
                let m_eff = (mi * mj) / (mi + mj);

                let dvx = vel[j3] - vxi;
                let dvy = vel[j3 + 1] - vyi;
                let dvz = vel[j3 + 2] - vzi;
                let vn = dvx * nx + dvy * ny + dvz * nz;

                let fn_damp = F::new(-2.0) * beta * F::sqrt(m_eff * kn) * vn;
                let fn_total = fn_spring + fn_damp;

                fxi -= fn_total * nx;
                fyi -= fn_total * ny;
                fzi -= fn_total * nz;
            }
        }

        force[i3] = fxi;
        force[i3 + 1] = fyi;
        force[i3 + 2] = fzi;
    }
}

// ── Slice launchers (dispatch on GpuRuntime) ────────────────────────────────

/// Toy SAXPY launcher: `y[i] = a * x[i] + y[i]`. Used to validate the cubecl
/// plumbing on each backend.
pub fn saxpy(rt: &GpuRuntime, a: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n == 0 {
        return;
    }
    match rt {
        GpuRuntime::Wgpu(client) => {
            let x_f32: Vec<f32> = x.iter().map(|v| *v as f32).collect();
            let y_f32: Vec<f32> = y.iter().map(|v| *v as f32).collect();
            let x_h = client.create_from_slice(f32::as_bytes(&x_f32));
            let y_h = client.create_from_slice(f32::as_bytes(&y_f32));
            unsafe {
                saxpy_kernel::launch_unchecked::<f32, cubecl::wgpu::WgpuRuntime>(
                    client,
                    cube_count_for(n),
                    CubeDim::new_1d(BLOCK),
                    a as f32,
                    ArrayArg::from_raw_parts(x_h, n),
                    ArrayArg::from_raw_parts(y_h.clone(), n),
                );
            }
            let bytes = client.read_one_unchecked(y_h);
            for (dst, src) in y.iter_mut().zip(f32::from_bytes(&bytes)) {
                *dst = *src as f64;
            }
        }
        GpuRuntime::Cpu(client) => {
            let x_h = client.create_from_slice(f64::as_bytes(x));
            let y_h = client.create_from_slice(f64::as_bytes(y));
            unsafe {
                saxpy_kernel::launch_unchecked::<f64, cubecl::cpu::CpuRuntime>(
                    client,
                    cube_count_for(n),
                    CubeDim::new_1d(BLOCK),
                    a,
                    ArrayArg::from_raw_parts(x_h, n),
                    ArrayArg::from_raw_parts(y_h.clone(), n),
                );
            }
            let bytes = client.read_one_unchecked(y_h);
            y.copy_from_slice(f64::from_bytes(&bytes));
        }
    }
}

/// Zero out a `[[f64; 3]]` buffer via the GPU. Round-trip pattern (upload →
/// kernel → download) — for actual perf, use `client.empty(...)` directly.
pub fn zero_force_buffer(rt: &GpuRuntime, forces: &mut [[f64; 3]]) {
    let n_floats = forces.len() * 3;
    if n_floats == 0 {
        return;
    }
    match rt {
        GpuRuntime::Wgpu(client) => {
            let flat = flatten_vec3_f32(forces);
            let h = client.create_from_slice(f32::as_bytes(&flat));
            unsafe {
                zero_kernel::launch_unchecked::<f32, cubecl::wgpu::WgpuRuntime>(
                    client,
                    cube_count_for(n_floats),
                    CubeDim::new_1d(BLOCK),
                    ArrayArg::from_raw_parts(h.clone(), n_floats),
                );
            }
            let bytes = client.read_one_unchecked(h);
            unflatten_vec3_f32(f32::from_bytes(&bytes), forces);
        }
        GpuRuntime::Cpu(client) => {
            let flat = flatten_vec3_f64(forces);
            let h = client.create_from_slice(f64::as_bytes(&flat));
            unsafe {
                zero_kernel::launch_unchecked::<f64, cubecl::cpu::CpuRuntime>(
                    client,
                    cube_count_for(n_floats),
                    CubeDim::new_1d(BLOCK),
                    ArrayArg::from_raw_parts(h.clone(), n_floats),
                );
            }
            let bytes = client.read_one_unchecked(h);
            unflatten_vec3_f64(f64::from_bytes(&bytes), forces);
        }
    }
}

/// Velocity-Verlet first half. Caller slices to `nlocal` particles; ghosts
/// are not integrated.
pub fn integrate_initial(
    rt: &GpuRuntime,
    dt: f64,
    inv_mass: &[f64],
    force: &[[f64; 3]],
    vel: &mut [[f64; 3]],
    pos: &mut [[f64; 3]],
) {
    let n = inv_mass.len();
    if n == 0 {
        return;
    }
    debug_assert_eq!(force.len(), n);
    debug_assert_eq!(vel.len(), n);
    debug_assert_eq!(pos.len(), n);

    match rt {
        GpuRuntime::Wgpu(client) => {
            let inv_mass_f32: Vec<f32> = inv_mass.iter().map(|x| *x as f32).collect();
            let force_f32 = flatten_vec3_f32(force);
            let vel_f32 = flatten_vec3_f32(vel);
            let pos_f32 = flatten_vec3_f32(pos);

            let inv_mass_h = client.create_from_slice(f32::as_bytes(&inv_mass_f32));
            let force_h = client.create_from_slice(f32::as_bytes(&force_f32));
            let vel_h = client.create_from_slice(f32::as_bytes(&vel_f32));
            let pos_h = client.create_from_slice(f32::as_bytes(&pos_f32));

            unsafe {
                integrate_initial_kernel::launch_unchecked::<f32, cubecl::wgpu::WgpuRuntime>(
                    client,
                    cube_count_for(n),
                    CubeDim::new_1d(BLOCK),
                    dt as f32,
                    ArrayArg::from_raw_parts(inv_mass_h, n),
                    ArrayArg::from_raw_parts(force_h, n * 3),
                    ArrayArg::from_raw_parts(vel_h.clone(), n * 3),
                    ArrayArg::from_raw_parts(pos_h.clone(), n * 3),
                );
            }

            let vel_bytes = client.read_one_unchecked(vel_h);
            let pos_bytes = client.read_one_unchecked(pos_h);
            unflatten_vec3_f32(f32::from_bytes(&vel_bytes), vel);
            unflatten_vec3_f32(f32::from_bytes(&pos_bytes), pos);
        }
        GpuRuntime::Cpu(client) => {
            let force_f64 = flatten_vec3_f64(force);
            let vel_f64 = flatten_vec3_f64(vel);
            let pos_f64 = flatten_vec3_f64(pos);

            let inv_mass_h = client.create_from_slice(f64::as_bytes(inv_mass));
            let force_h = client.create_from_slice(f64::as_bytes(&force_f64));
            let vel_h = client.create_from_slice(f64::as_bytes(&vel_f64));
            let pos_h = client.create_from_slice(f64::as_bytes(&pos_f64));

            unsafe {
                integrate_initial_kernel::launch_unchecked::<f64, cubecl::cpu::CpuRuntime>(
                    client,
                    cube_count_for(n),
                    CubeDim::new_1d(BLOCK),
                    dt,
                    ArrayArg::from_raw_parts(inv_mass_h, n),
                    ArrayArg::from_raw_parts(force_h, n * 3),
                    ArrayArg::from_raw_parts(vel_h.clone(), n * 3),
                    ArrayArg::from_raw_parts(pos_h.clone(), n * 3),
                );
            }

            let vel_bytes = client.read_one_unchecked(vel_h);
            let pos_bytes = client.read_one_unchecked(pos_h);
            unflatten_vec3_f64(f64::from_bytes(&vel_bytes), vel);
            unflatten_vec3_f64(f64::from_bytes(&pos_bytes), pos);
        }
    }
}

/// Velocity-Verlet second half.
pub fn integrate_final(
    rt: &GpuRuntime,
    dt: f64,
    inv_mass: &[f64],
    force: &[[f64; 3]],
    vel: &mut [[f64; 3]],
) {
    let n = inv_mass.len();
    if n == 0 {
        return;
    }
    debug_assert_eq!(force.len(), n);
    debug_assert_eq!(vel.len(), n);

    match rt {
        GpuRuntime::Wgpu(client) => {
            let inv_mass_f32: Vec<f32> = inv_mass.iter().map(|x| *x as f32).collect();
            let force_f32 = flatten_vec3_f32(force);
            let vel_f32 = flatten_vec3_f32(vel);

            let inv_mass_h = client.create_from_slice(f32::as_bytes(&inv_mass_f32));
            let force_h = client.create_from_slice(f32::as_bytes(&force_f32));
            let vel_h = client.create_from_slice(f32::as_bytes(&vel_f32));

            unsafe {
                integrate_final_kernel::launch_unchecked::<f32, cubecl::wgpu::WgpuRuntime>(
                    client,
                    cube_count_for(n),
                    CubeDim::new_1d(BLOCK),
                    dt as f32,
                    ArrayArg::from_raw_parts(inv_mass_h, n),
                    ArrayArg::from_raw_parts(force_h, n * 3),
                    ArrayArg::from_raw_parts(vel_h.clone(), n * 3),
                );
            }

            let vel_bytes = client.read_one_unchecked(vel_h);
            unflatten_vec3_f32(f32::from_bytes(&vel_bytes), vel);
        }
        GpuRuntime::Cpu(client) => {
            let force_f64 = flatten_vec3_f64(force);
            let vel_f64 = flatten_vec3_f64(vel);

            let inv_mass_h = client.create_from_slice(f64::as_bytes(inv_mass));
            let force_h = client.create_from_slice(f64::as_bytes(&force_f64));
            let vel_h = client.create_from_slice(f64::as_bytes(&vel_f64));

            unsafe {
                integrate_final_kernel::launch_unchecked::<f64, cubecl::cpu::CpuRuntime>(
                    client,
                    cube_count_for(n),
                    CubeDim::new_1d(BLOCK),
                    dt,
                    ArrayArg::from_raw_parts(inv_mass_h, n),
                    ArrayArg::from_raw_parts(force_h, n * 3),
                    ArrayArg::from_raw_parts(vel_h.clone(), n * 3),
                );
            }

            let vel_bytes = client.read_one_unchecked(vel_h);
            unflatten_vec3_f64(f64::from_bytes(&vel_bytes), vel);
        }
    }
}

/// Quaternion-Verlet first half.
pub fn rotate_initial(
    rt: &GpuRuntime,
    dt: f64,
    inv_inertia: &[f64],
    torque: &[[f64; 3]],
    omega: &mut [[f64; 3]],
    quaternion: &mut [[f64; 4]],
) {
    let n = inv_inertia.len();
    if n == 0 {
        return;
    }
    debug_assert_eq!(torque.len(), n);
    debug_assert_eq!(omega.len(), n);
    debug_assert_eq!(quaternion.len(), n);

    match rt {
        GpuRuntime::Wgpu(client) => {
            let inv_inertia_f32: Vec<f32> = inv_inertia.iter().map(|x| *x as f32).collect();
            let torque_f32 = flatten_vec3_f32(torque);
            let omega_f32 = flatten_vec3_f32(omega);
            let quat_f32 = flatten_vec4_f32(quaternion);

            let inv_h = client.create_from_slice(f32::as_bytes(&inv_inertia_f32));
            let tor_h = client.create_from_slice(f32::as_bytes(&torque_f32));
            let om_h = client.create_from_slice(f32::as_bytes(&omega_f32));
            let q_h = client.create_from_slice(f32::as_bytes(&quat_f32));

            unsafe {
                rotate_initial_kernel::launch_unchecked::<f32, cubecl::wgpu::WgpuRuntime>(
                    client,
                    cube_count_for(n),
                    CubeDim::new_1d(BLOCK),
                    dt as f32,
                    ArrayArg::from_raw_parts(inv_h, n),
                    ArrayArg::from_raw_parts(tor_h, n * 3),
                    ArrayArg::from_raw_parts(om_h.clone(), n * 3),
                    ArrayArg::from_raw_parts(q_h.clone(), n * 4),
                );
            }

            let om_bytes = client.read_one_unchecked(om_h);
            let q_bytes = client.read_one_unchecked(q_h);
            unflatten_vec3_f32(f32::from_bytes(&om_bytes), omega);
            unflatten_vec4_f32(f32::from_bytes(&q_bytes), quaternion);
        }
        GpuRuntime::Cpu(client) => {
            let torque_f64 = flatten_vec3_f64(torque);
            let omega_f64 = flatten_vec3_f64(omega);
            let quat_f64 = flatten_vec4_f64(quaternion);

            let inv_h = client.create_from_slice(f64::as_bytes(inv_inertia));
            let tor_h = client.create_from_slice(f64::as_bytes(&torque_f64));
            let om_h = client.create_from_slice(f64::as_bytes(&omega_f64));
            let q_h = client.create_from_slice(f64::as_bytes(&quat_f64));

            unsafe {
                rotate_initial_kernel::launch_unchecked::<f64, cubecl::cpu::CpuRuntime>(
                    client,
                    cube_count_for(n),
                    CubeDim::new_1d(BLOCK),
                    dt,
                    ArrayArg::from_raw_parts(inv_h, n),
                    ArrayArg::from_raw_parts(tor_h, n * 3),
                    ArrayArg::from_raw_parts(om_h.clone(), n * 3),
                    ArrayArg::from_raw_parts(q_h.clone(), n * 4),
                );
            }

            let om_bytes = client.read_one_unchecked(om_h);
            let q_bytes = client.read_one_unchecked(q_h);
            unflatten_vec3_f64(f64::from_bytes(&om_bytes), omega);
            unflatten_vec4_f64(f64::from_bytes(&q_bytes), quaternion);
        }
    }
}

/// Quaternion-Verlet second half.
pub fn rotate_final(
    rt: &GpuRuntime,
    dt: f64,
    inv_inertia: &[f64],
    torque: &[[f64; 3]],
    omega: &mut [[f64; 3]],
) {
    let n = inv_inertia.len();
    if n == 0 {
        return;
    }
    debug_assert_eq!(torque.len(), n);
    debug_assert_eq!(omega.len(), n);

    match rt {
        GpuRuntime::Wgpu(client) => {
            let inv_inertia_f32: Vec<f32> = inv_inertia.iter().map(|x| *x as f32).collect();
            let torque_f32 = flatten_vec3_f32(torque);
            let omega_f32 = flatten_vec3_f32(omega);

            let inv_h = client.create_from_slice(f32::as_bytes(&inv_inertia_f32));
            let tor_h = client.create_from_slice(f32::as_bytes(&torque_f32));
            let om_h = client.create_from_slice(f32::as_bytes(&omega_f32));

            unsafe {
                rotate_final_kernel::launch_unchecked::<f32, cubecl::wgpu::WgpuRuntime>(
                    client,
                    cube_count_for(n),
                    CubeDim::new_1d(BLOCK),
                    dt as f32,
                    ArrayArg::from_raw_parts(inv_h, n),
                    ArrayArg::from_raw_parts(tor_h, n * 3),
                    ArrayArg::from_raw_parts(om_h.clone(), n * 3),
                );
            }

            let om_bytes = client.read_one_unchecked(om_h);
            unflatten_vec3_f32(f32::from_bytes(&om_bytes), omega);
        }
        GpuRuntime::Cpu(client) => {
            let torque_f64 = flatten_vec3_f64(torque);
            let omega_f64 = flatten_vec3_f64(omega);

            let inv_h = client.create_from_slice(f64::as_bytes(inv_inertia));
            let tor_h = client.create_from_slice(f64::as_bytes(&torque_f64));
            let om_h = client.create_from_slice(f64::as_bytes(&omega_f64));

            unsafe {
                rotate_final_kernel::launch_unchecked::<f64, cubecl::cpu::CpuRuntime>(
                    client,
                    cube_count_for(n),
                    CubeDim::new_1d(BLOCK),
                    dt,
                    ArrayArg::from_raw_parts(inv_h, n),
                    ArrayArg::from_raw_parts(tor_h, n * 3),
                    ArrayArg::from_raw_parts(om_h.clone(), n * 3),
                );
            }

            let om_bytes = client.read_one_unchecked(om_h);
            unflatten_vec3_f64(f64::from_bytes(&om_bytes), omega);
        }
    }
}

// ── Resident-buffer launchers ───────────────────────────────────────────────
//
// These take cubecl `Handle`s directly (no upload/download per call). Used
// by the v2 GPU systems that work against [`AtomGpu`] / [`DemAtomGpu`].

use cubecl::server::Handle;

/// Zero out a resident GPU buffer of `n_floats` scalar elements.
pub fn zero_resident(rt: &GpuRuntime, buf: &Handle, n_floats: usize) {
    if n_floats == 0 {
        return;
    }
    match rt {
        GpuRuntime::Wgpu(client) => unsafe {
            zero_kernel::launch_unchecked::<f32, cubecl::wgpu::WgpuRuntime>(
                client,
                cube_count_for(n_floats),
                CubeDim::new_1d(BLOCK),
                ArrayArg::from_raw_parts(buf.clone(), n_floats),
            );
        },
        GpuRuntime::Cpu(client) => unsafe {
            zero_kernel::launch_unchecked::<f64, cubecl::cpu::CpuRuntime>(
                client,
                cube_count_for(n_floats),
                CubeDim::new_1d(BLOCK),
                ArrayArg::from_raw_parts(buf.clone(), n_floats),
            );
        },
    }
}

/// Velocity-Verlet first half on resident GPU buffers. `nlocal` is the
/// number of particles; vec3 buffers must be sized `3 * nlocal`.
pub fn integrate_initial_resident(
    rt: &GpuRuntime,
    dt: f64,
    inv_mass: &Handle,
    force: &Handle,
    vel: &Handle,
    pos: &Handle,
    nlocal: usize,
) {
    if nlocal == 0 {
        return;
    }
    match rt {
        GpuRuntime::Wgpu(client) => unsafe {
            integrate_initial_kernel::launch_unchecked::<f32, cubecl::wgpu::WgpuRuntime>(
                client,
                cube_count_for(nlocal),
                CubeDim::new_1d(BLOCK),
                dt as f32,
                ArrayArg::from_raw_parts(inv_mass.clone(), nlocal),
                ArrayArg::from_raw_parts(force.clone(), nlocal * 3),
                ArrayArg::from_raw_parts(vel.clone(), nlocal * 3),
                ArrayArg::from_raw_parts(pos.clone(), nlocal * 3),
            );
        },
        GpuRuntime::Cpu(client) => unsafe {
            integrate_initial_kernel::launch_unchecked::<f64, cubecl::cpu::CpuRuntime>(
                client,
                cube_count_for(nlocal),
                CubeDim::new_1d(BLOCK),
                dt,
                ArrayArg::from_raw_parts(inv_mass.clone(), nlocal),
                ArrayArg::from_raw_parts(force.clone(), nlocal * 3),
                ArrayArg::from_raw_parts(vel.clone(), nlocal * 3),
                ArrayArg::from_raw_parts(pos.clone(), nlocal * 3),
            );
        },
    }
}

/// Velocity-Verlet second half on resident GPU buffers.
pub fn integrate_final_resident(
    rt: &GpuRuntime,
    dt: f64,
    inv_mass: &Handle,
    force: &Handle,
    vel: &Handle,
    nlocal: usize,
) {
    if nlocal == 0 {
        return;
    }
    match rt {
        GpuRuntime::Wgpu(client) => unsafe {
            integrate_final_kernel::launch_unchecked::<f32, cubecl::wgpu::WgpuRuntime>(
                client,
                cube_count_for(nlocal),
                CubeDim::new_1d(BLOCK),
                dt as f32,
                ArrayArg::from_raw_parts(inv_mass.clone(), nlocal),
                ArrayArg::from_raw_parts(force.clone(), nlocal * 3),
                ArrayArg::from_raw_parts(vel.clone(), nlocal * 3),
            );
        },
        GpuRuntime::Cpu(client) => unsafe {
            integrate_final_kernel::launch_unchecked::<f64, cubecl::cpu::CpuRuntime>(
                client,
                cube_count_for(nlocal),
                CubeDim::new_1d(BLOCK),
                dt,
                ArrayArg::from_raw_parts(inv_mass.clone(), nlocal),
                ArrayArg::from_raw_parts(force.clone(), nlocal * 3),
                ArrayArg::from_raw_parts(vel.clone(), nlocal * 3),
            );
        },
    }
}

/// Quaternion-Verlet first half on resident GPU buffers.
pub fn rotate_initial_resident(
    rt: &GpuRuntime,
    dt: f64,
    inv_inertia: &Handle,
    torque: &Handle,
    omega: &Handle,
    quaternion: &Handle,
    nlocal: usize,
) {
    if nlocal == 0 {
        return;
    }
    match rt {
        GpuRuntime::Wgpu(client) => unsafe {
            rotate_initial_kernel::launch_unchecked::<f32, cubecl::wgpu::WgpuRuntime>(
                client,
                cube_count_for(nlocal),
                CubeDim::new_1d(BLOCK),
                dt as f32,
                ArrayArg::from_raw_parts(inv_inertia.clone(), nlocal),
                ArrayArg::from_raw_parts(torque.clone(), nlocal * 3),
                ArrayArg::from_raw_parts(omega.clone(), nlocal * 3),
                ArrayArg::from_raw_parts(quaternion.clone(), nlocal * 4),
            );
        },
        GpuRuntime::Cpu(client) => unsafe {
            rotate_initial_kernel::launch_unchecked::<f64, cubecl::cpu::CpuRuntime>(
                client,
                cube_count_for(nlocal),
                CubeDim::new_1d(BLOCK),
                dt,
                ArrayArg::from_raw_parts(inv_inertia.clone(), nlocal),
                ArrayArg::from_raw_parts(torque.clone(), nlocal * 3),
                ArrayArg::from_raw_parts(omega.clone(), nlocal * 3),
                ArrayArg::from_raw_parts(quaternion.clone(), nlocal * 4),
            );
        },
    }
}

/// Hertz-only normal contact force on resident GPU buffers. Single material.
///
/// `n_local` is the number of local atoms (gets a force write per thread).
/// `pos`, `vel`, `radius`, `inv_mass` should be sized for **all atoms**
/// (local + ghost) since neighbor indices reference into those buffers.
/// `neighbor_offsets` is sized `n_local + 1`, `neighbor_indices` is the flat
/// CSR neighbor list.
pub fn hertz_normal_resident(
    rt: &GpuRuntime,
    e_eff: f64,
    beta: f64,
    pos: &Handle,
    vel: &Handle,
    radius: &Handle,
    inv_mass: &Handle,
    neighbor_offsets: &Handle,
    neighbor_indices: &Handle,
    force: &Handle,
    n_local: usize,
    n_neighbor_indices: usize,
) {
    if n_local == 0 {
        return;
    }
    match rt {
        GpuRuntime::Wgpu(client) => unsafe {
            hertz_normal_kernel::launch_unchecked::<f32, cubecl::wgpu::WgpuRuntime>(
                client,
                cube_count_for(n_local),
                CubeDim::new_1d(BLOCK),
                e_eff as f32,
                beta as f32,
                ArrayArg::from_raw_parts(pos.clone(), n_local * 3),
                ArrayArg::from_raw_parts(vel.clone(), n_local * 3),
                ArrayArg::from_raw_parts(radius.clone(), n_local),
                ArrayArg::from_raw_parts(inv_mass.clone(), n_local),
                ArrayArg::from_raw_parts(neighbor_offsets.clone(), n_local + 1),
                ArrayArg::from_raw_parts(neighbor_indices.clone(), n_neighbor_indices),
                ArrayArg::from_raw_parts(force.clone(), n_local * 3),
            );
        },
        GpuRuntime::Cpu(client) => unsafe {
            hertz_normal_kernel::launch_unchecked::<f64, cubecl::cpu::CpuRuntime>(
                client,
                cube_count_for(n_local),
                CubeDim::new_1d(BLOCK),
                e_eff,
                beta,
                ArrayArg::from_raw_parts(pos.clone(), n_local * 3),
                ArrayArg::from_raw_parts(vel.clone(), n_local * 3),
                ArrayArg::from_raw_parts(radius.clone(), n_local),
                ArrayArg::from_raw_parts(inv_mass.clone(), n_local),
                ArrayArg::from_raw_parts(neighbor_offsets.clone(), n_local + 1),
                ArrayArg::from_raw_parts(neighbor_indices.clone(), n_neighbor_indices),
                ArrayArg::from_raw_parts(force.clone(), n_local * 3),
            );
        },
    }
}

/// Quaternion-Verlet second half on resident GPU buffers.
pub fn rotate_final_resident(
    rt: &GpuRuntime,
    dt: f64,
    inv_inertia: &Handle,
    torque: &Handle,
    omega: &Handle,
    nlocal: usize,
) {
    if nlocal == 0 {
        return;
    }
    match rt {
        GpuRuntime::Wgpu(client) => unsafe {
            rotate_final_kernel::launch_unchecked::<f32, cubecl::wgpu::WgpuRuntime>(
                client,
                cube_count_for(nlocal),
                CubeDim::new_1d(BLOCK),
                dt as f32,
                ArrayArg::from_raw_parts(inv_inertia.clone(), nlocal),
                ArrayArg::from_raw_parts(torque.clone(), nlocal * 3),
                ArrayArg::from_raw_parts(omega.clone(), nlocal * 3),
            );
        },
        GpuRuntime::Cpu(client) => unsafe {
            rotate_final_kernel::launch_unchecked::<f64, cubecl::cpu::CpuRuntime>(
                client,
                cube_count_for(nlocal),
                CubeDim::new_1d(BLOCK),
                dt,
                ArrayArg::from_raw_parts(inv_inertia.clone(), nlocal),
                ArrayArg::from_raw_parts(torque.clone(), nlocal * 3),
                ArrayArg::from_raw_parts(omega.clone(), nlocal * 3),
            );
        },
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_forces_f32_even_when_f64_requested() {
        let p = select_precision(BackendKind::Wgpu, Some(Precision::F64));
        assert_eq!(p, Precision::F32);
    }

    #[test]
    fn cuda_honors_f64_request() {
        let p = select_precision(BackendKind::Cuda, Some(Precision::F64));
        assert_eq!(p, Precision::F64);
    }

    #[test]
    fn saxpy_runs_on_both_backends() {
        for backend in [BackendKind::Wgpu, BackendKind::Cpu] {
            let rt = GpuRuntime::new(backend);
            let x = vec![1.0_f64, 2.0, 3.0, 4.0];
            let mut y = vec![10.0_f64, 20.0, 30.0, 40.0];
            saxpy(&rt, 2.0, &x, &mut y);
            for (got, want) in y.iter().zip([12.0, 24.0, 36.0, 48.0]) {
                assert!((got - want).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn zero_force_buffer_round_trips_on_both_backends() {
        for backend in [BackendKind::Wgpu, BackendKind::Cpu] {
            let rt = GpuRuntime::new(backend);
            let mut forces: Vec<[f64; 3]> = vec![[1.0, 2.0, 3.0]; 1024];
            zero_force_buffer(&rt, &mut forces);
            for f in &forces {
                assert_eq!(*f, [0.0, 0.0, 0.0]);
            }
        }
    }

    fn integrate_initial_cpu_reference(
        dt: f64,
        inv_mass: &[f64],
        force: &[[f64; 3]],
        vel: &mut [[f64; 3]],
        pos: &mut [[f64; 3]],
    ) {
        for i in 0..inv_mass.len() {
            let half_dt_over_m = 0.5 * dt * inv_mass[i];
            let f = force[i];
            let v = &mut vel[i];
            v[0] += half_dt_over_m * f[0];
            v[1] += half_dt_over_m * f[1];
            v[2] += half_dt_over_m * f[2];
            let p = &mut pos[i];
            p[0] += v[0] * dt;
            p[1] += v[1] * dt;
            p[2] += v[2] * dt;
        }
    }

    fn integrate_final_cpu_reference(
        dt: f64,
        inv_mass: &[f64],
        force: &[[f64; 3]],
        vel: &mut [[f64; 3]],
    ) {
        for i in 0..inv_mass.len() {
            let half_dt_over_m = 0.5 * dt * inv_mass[i];
            let f = force[i];
            let v = &mut vel[i];
            v[0] += half_dt_over_m * f[0];
            v[1] += half_dt_over_m * f[1];
            v[2] += half_dt_over_m * f[2];
        }
    }

    fn integrate_fixture(n: usize) -> (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let mut inv_mass = Vec::with_capacity(n);
        let mut force = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        let mut pos = Vec::with_capacity(n);
        for i in 0..n {
            inv_mass.push(1.0 / (1.0 + (i % 5) as f64));
            let s = (i + 1) as f64;
            force.push([0.1 * s, -0.2 * s, 0.05 * s]);
            vel.push([0.5, -0.3, 0.7]);
            pos.push([s * 0.01, s * 0.02, s * 0.03]);
        }
        (inv_mass, force, vel, pos)
    }

    fn assert_close_vec3(got: &[[f64; 3]], want: &[[f64; 3]], tol: f64) {
        for i in 0..got.len() {
            for d in 0..3 {
                assert!(
                    (got[i][d] - want[i][d]).abs() < tol,
                    "[{i}][{d}]: got={} want={}",
                    got[i][d],
                    want[i][d],
                );
            }
        }
    }

    fn assert_close_vec4(got: &[[f64; 4]], want: &[[f64; 4]], tol: f64) {
        for i in 0..got.len() {
            for d in 0..4 {
                assert!((got[i][d] - want[i][d]).abs() < tol);
            }
        }
    }

    #[test]
    fn integrate_initial_matches_reference() {
        // f64 (CPU runtime) should match bit-for-bit; f32 (wgpu) within 1e-5.
        for (backend, tol) in [(BackendKind::Cpu, 1e-15), (BackendKind::Wgpu, 1e-5)] {
            let rt = GpuRuntime::new(backend);
            let (inv_mass, force, mut vel, mut pos) = integrate_fixture(512);
            let mut vel_ref = vel.clone();
            let mut pos_ref = pos.clone();
            let dt = 1e-6;

            integrate_initial(&rt, dt, &inv_mass, &force, &mut vel, &mut pos);
            integrate_initial_cpu_reference(dt, &inv_mass, &force, &mut vel_ref, &mut pos_ref);

            assert_close_vec3(&vel, &vel_ref, tol);
            assert_close_vec3(&pos, &pos_ref, tol);
        }
    }

    #[test]
    fn integrate_final_matches_reference() {
        for (backend, tol) in [(BackendKind::Cpu, 1e-15), (BackendKind::Wgpu, 1e-5)] {
            let rt = GpuRuntime::new(backend);
            let (inv_mass, force, mut vel, _) = integrate_fixture(512);
            let mut vel_ref = vel.clone();
            let dt = 1e-6;

            integrate_final(&rt, dt, &inv_mass, &force, &mut vel);
            integrate_final_cpu_reference(dt, &inv_mass, &force, &mut vel_ref);

            assert_close_vec3(&vel, &vel_ref, tol);
        }
    }

    fn rotate_initial_cpu_reference(
        dt: f64,
        inv_inertia: &[f64],
        torque: &[[f64; 3]],
        omega: &mut [[f64; 3]],
        quaternion: &mut [[f64; 4]],
    ) {
        for i in 0..inv_inertia.len() {
            let inv_i = inv_inertia[i];
            if inv_i == 0.0 {
                continue;
            }
            omega[i][0] += 0.5 * dt * torque[i][0] * inv_i;
            omega[i][1] += 0.5 * dt * torque[i][1] * inv_i;
            omega[i][2] += 0.5 * dt * torque[i][2] * inv_i;

            let (ox, oy, oz) = (omega[i][0], omega[i][1], omega[i][2]);
            let mag = (ox * ox + oy * oy + oz * oz).sqrt();
            let angle = mag * dt;
            if angle > 1e-14 {
                let inv = 1.0 / mag;
                let (ax, ay, az) = (ox * inv, oy * inv, oz * inv);
                let half = angle * 0.5;
                let s = half.sin();
                let dq = [half.cos(), ax * s, ay * s, az * s];
                let q = quaternion[i];
                quaternion[i] = [
                    dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2] - dq[3] * q[3],
                    dq[0] * q[1] + dq[1] * q[0] + dq[2] * q[3] - dq[3] * q[2],
                    dq[0] * q[2] - dq[1] * q[3] + dq[2] * q[0] + dq[3] * q[1],
                    dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1] + dq[3] * q[0],
                ];
            }
        }
    }

    fn rotate_final_cpu_reference(
        dt: f64,
        inv_inertia: &[f64],
        torque: &[[f64; 3]],
        omega: &mut [[f64; 3]],
    ) {
        for i in 0..inv_inertia.len() {
            let inv_i = inv_inertia[i];
            if inv_i == 0.0 {
                continue;
            }
            omega[i][0] += 0.5 * dt * torque[i][0] * inv_i;
            omega[i][1] += 0.5 * dt * torque[i][1] * inv_i;
            omega[i][2] += 0.5 * dt * torque[i][2] * inv_i;
        }
    }

    fn rot_fixture(n: usize) -> (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<[f64; 4]>) {
        let mut inv_inertia = Vec::with_capacity(n);
        let mut torque = Vec::with_capacity(n);
        let mut omega = Vec::with_capacity(n);
        let mut quat = Vec::with_capacity(n);
        for i in 0..n {
            inv_inertia.push(if i % 7 == 0 { 0.0 } else { 1.0 / (1.0 + (i % 5) as f64) });
            let s = (i + 1) as f64;
            torque.push([0.01 * s, -0.02 * s, 0.005 * s]);
            omega.push([0.1, -0.05, 0.2]);
            quat.push([1.0, 0.0, 0.0, 0.0]);
        }
        (inv_inertia, torque, omega, quat)
    }

    #[test]
    fn rotate_initial_matches_reference() {
        for (backend, tol) in [(BackendKind::Cpu, 1e-14), (BackendKind::Wgpu, 1e-5)] {
            let rt = GpuRuntime::new(backend);
            let (inv_inertia, torque, mut omega, mut quat) = rot_fixture(512);
            let mut omega_ref = omega.clone();
            let mut quat_ref = quat.clone();
            let dt = 1e-5;

            rotate_initial(&rt, dt, &inv_inertia, &torque, &mut omega, &mut quat);
            rotate_initial_cpu_reference(
                dt,
                &inv_inertia,
                &torque,
                &mut omega_ref,
                &mut quat_ref,
            );

            assert_close_vec3(&omega, &omega_ref, tol);
            assert_close_vec4(&quat, &quat_ref, tol);
        }
    }

    #[test]
    fn rotate_final_matches_reference() {
        for (backend, tol) in [(BackendKind::Cpu, 1e-15), (BackendKind::Wgpu, 1e-5)] {
            let rt = GpuRuntime::new(backend);
            let (inv_inertia, torque, mut omega, _) = rot_fixture(512);
            let mut omega_ref = omega.clone();
            let dt = 1e-5;

            rotate_final(&rt, dt, &inv_inertia, &torque, &mut omega);
            rotate_final_cpu_reference(dt, &inv_inertia, &torque, &mut omega_ref);

            assert_close_vec3(&omega, &omega_ref, tol);
        }
    }

    /// GpuField vec3 round-trip: upload then download must yield input.
    #[test]
    fn gpu_field_vec3_round_trips_on_both_backends() {
        for (backend, tol) in [(BackendKind::Cpu, 1e-15), (BackendKind::Wgpu, 1e-5)] {
            let rt = GpuRuntime::new(backend);
            let mut field = GpuField::new();
            let input: Vec<[f64; 3]> = (0..256)
                .map(|i| [i as f64 * 0.1, i as f64 * -0.2, i as f64 * 0.3])
                .collect();
            field.upload_vec3(&input, &rt);
            let mut output = vec![[0.0_f64; 3]; input.len()];
            field.download_vec3(&mut output, &rt);
            assert_close_vec3(&output, &input, tol);
        }
    }

    /// Hertz-only CPU reference. Full-pair iteration: each atom sums over its
    /// neighbor list, so each pair contributes independently to both atoms.
    fn hertz_normal_cpu_reference(
        e_eff: f64,
        beta: f64,
        pos: &[[f64; 3]],
        vel: &[[f64; 3]],
        radius: &[f64],
        inv_mass: &[f64],
        neighbor_offsets: &[u32],
        neighbor_indices: &[u32],
        force: &mut [[f64; 3]],
    ) {
        let n = neighbor_offsets.len() - 1;
        for i in 0..n {
            let mi = if inv_mass[i] > 0.0 { 1.0 / inv_mass[i] } else { 1e30 };
            let mut fi = [0.0_f64; 3];
            let start = neighbor_offsets[i] as usize;
            let end = neighbor_offsets[i + 1] as usize;
            for nidx in start..end {
                let j = neighbor_indices[nidx] as usize;
                let dx = pos[j][0] - pos[i][0];
                let dy = pos[j][1] - pos[i][1];
                let dz = pos[j][2] - pos[i][2];
                let r2 = dx * dx + dy * dy + dz * dz;
                let r_sum = radius[i] + radius[j];
                if r2 < r_sum * r_sum && r2 > 0.0 {
                    let dist = r2.sqrt();
                    let nx = dx / dist;
                    let ny = dy / dist;
                    let nz = dz / dist;
                    let delta = r_sum - dist;
                    let r_eff = (radius[i] * radius[j]) / r_sum;
                    let kn = (4.0 / 3.0) * e_eff * (r_eff * delta).sqrt();
                    let fn_spring = kn * delta;
                    let mj = if inv_mass[j] > 0.0 { 1.0 / inv_mass[j] } else { 1e30 };
                    let m_eff = (mi * mj) / (mi + mj);
                    let dvx = vel[j][0] - vel[i][0];
                    let dvy = vel[j][1] - vel[i][1];
                    let dvz = vel[j][2] - vel[i][2];
                    let vn = dvx * nx + dvy * ny + dvz * nz;
                    let fn_damp = -2.0 * beta * (m_eff * kn).sqrt() * vn;
                    let fn_total = fn_spring + fn_damp;
                    fi[0] -= fn_total * nx;
                    fi[1] -= fn_total * ny;
                    fi[2] -= fn_total * nz;
                }
            }
            force[i] = fi;
        }
    }

    /// Two atoms in head-on Hertzian contact with damping. Validates the
    /// kernel against the CPU reference. f64 path matches bit-for-bit; f32
    /// within material-property-scaled tolerance.
    #[test]
    fn hertz_normal_two_atoms_matches_reference() {
        // E* = 8.7e9 (glass), beta = 0.1, particles overlap by 5%.
        let e_eff = 8.7e9_f64;
        let beta = 0.1_f64;
        let pos: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [1.5e-3, 0.0, 0.0]];
        let vel: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [-0.1, 0.0, 0.0]];
        let radius = vec![1.0e-3_f64, 1.0e-3];
        let inv_mass = vec![1.0_f64 / 1.0e-5, 1.0 / 1.0e-5];
        // Full-pair neighbor list.
        let neighbor_offsets = vec![0_u32, 1, 2];
        let neighbor_indices = vec![1_u32, 0];

        // CPU reference.
        let mut force_ref = vec![[0.0_f64; 3]; 2];
        hertz_normal_cpu_reference(
            e_eff,
            beta,
            &pos,
            &vel,
            &radius,
            &inv_mass,
            &neighbor_offsets,
            &neighbor_indices,
            &mut force_ref,
        );

        // The two atoms should get equal-and-opposite force.
        for d in 0..3 {
            assert!((force_ref[0][d] + force_ref[1][d]).abs() < 1e-6,
                "Newton-3: f0[{d}] + f1[{d}] = {} should be 0",
                force_ref[0][d] + force_ref[1][d]);
        }

        for (backend, tol_ratio) in [(BackendKind::Cpu, 1e-12), (BackendKind::Wgpu, 1e-3)] {
            let rt = GpuRuntime::new(backend);

            // Upload input buffers.
            let pos_h = upload_vec3_handle(&pos, &rt);
            let vel_h = upload_vec3_handle(&vel, &rt);
            let radius_h = upload_scalar_handle(&radius, &rt);
            let inv_mass_h = upload_scalar_handle(&inv_mass, &rt);
            let offsets_h = upload_u32_handle(&neighbor_offsets, &rt);
            let indices_h = upload_u32_handle(&neighbor_indices, &rt);
            let force_h = match &rt {
                GpuRuntime::Wgpu(c) => {
                    let zeros: Vec<f32> = vec![0.0; 6];
                    c.create_from_slice(f32::as_bytes(&zeros))
                }
                GpuRuntime::Cpu(c) => {
                    let zeros: Vec<f64> = vec![0.0; 6];
                    c.create_from_slice(f64::as_bytes(&zeros))
                }
            };

            hertz_normal_resident(
                &rt,
                e_eff,
                beta,
                &pos_h,
                &vel_h,
                &radius_h,
                &inv_mass_h,
                &offsets_h,
                &indices_h,
                &force_h,
                2,
                2,
            );

            let mut force_gpu = vec![[0.0_f64; 3]; 2];
            match &rt {
                GpuRuntime::Wgpu(c) => {
                    let bytes = c.read_one_unchecked(force_h);
                    unflatten_vec3_f32(f32::from_bytes(&bytes), &mut force_gpu);
                }
                GpuRuntime::Cpu(c) => {
                    let bytes = c.read_one_unchecked(force_h);
                    unflatten_vec3_f64(f64::from_bytes(&bytes), &mut force_gpu);
                }
            }

            // Magnitude is large (Hertz with E=8.7e9 produces big forces);
            // use a relative tolerance.
            let scale = force_ref[0][0].abs().max(1.0);
            for i in 0..2 {
                for d in 0..3 {
                    let err = (force_gpu[i][d] - force_ref[i][d]).abs();
                    assert!(
                        err < tol_ratio * scale,
                        "{:?}: force[{i}][{d}]: gpu={} ref={} err={}",
                        backend, force_gpu[i][d], force_ref[i][d], err
                    );
                }
            }
        }
    }

    fn upload_vec3_handle(data: &[[f64; 3]], rt: &GpuRuntime) -> cubecl::server::Handle {
        match rt {
            GpuRuntime::Wgpu(c) => c.create_from_slice(f32::as_bytes(&flatten_vec3_f32(data))),
            GpuRuntime::Cpu(c) => c.create_from_slice(f64::as_bytes(&flatten_vec3_f64(data))),
        }
    }

    fn upload_scalar_handle(data: &[f64], rt: &GpuRuntime) -> cubecl::server::Handle {
        match rt {
            GpuRuntime::Wgpu(c) => {
                let f32_data: Vec<f32> = data.iter().map(|x| *x as f32).collect();
                c.create_from_slice(f32::as_bytes(&f32_data))
            }
            GpuRuntime::Cpu(c) => c.create_from_slice(f64::as_bytes(data)),
        }
    }

    fn upload_u32_handle(data: &[u32], rt: &GpuRuntime) -> cubecl::server::Handle {
        match rt {
            GpuRuntime::Wgpu(c) => c.create_from_slice(u32::as_bytes(data)),
            GpuRuntime::Cpu(c) => c.create_from_slice(u32::as_bytes(data)),
        }
    }

    /// Full resident-buffer integration: upload, run integrate_initial_resident,
    /// download. Compare against the slice launcher (which we already validated).
    #[test]
    fn integrate_initial_resident_matches_slice_launcher() {
        for (backend, tol) in [(BackendKind::Cpu, 1e-15), (BackendKind::Wgpu, 1e-5)] {
            let rt = GpuRuntime::new(backend);
            let (inv_mass, force, mut vel_a, mut pos_a) = integrate_fixture(512);
            let mut vel_b = vel_a.clone();
            let mut pos_b = pos_a.clone();
            let dt = 1e-6;

            // Path A: slice launcher (uploads + downloads internally).
            integrate_initial(&rt, dt, &inv_mass, &force, &mut vel_a, &mut pos_a);

            // Path B: AtomGpu + resident launcher.
            let mut gpu = AtomGpu::default();
            gpu.inv_mass.upload_scalar(&inv_mass, &rt);
            gpu.force.upload_vec3(&force, &rt);
            gpu.vel.upload_vec3(&vel_b, &rt);
            gpu.pos.upload_vec3(&pos_b, &rt);
            integrate_initial_resident(
                &rt,
                dt,
                gpu.inv_mass.handle(),
                gpu.force.handle(),
                gpu.vel.handle(),
                gpu.pos.handle(),
                512,
            );
            gpu.vel.download_vec3(&mut vel_b, &rt);
            gpu.pos.download_vec3(&mut pos_b, &rt);

            assert_close_vec3(&vel_a, &vel_b, tol);
            assert_close_vec3(&pos_a, &pos_b, tol);
        }
    }
}

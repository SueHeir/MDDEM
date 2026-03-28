//! Rigid body data structures, inertia tensor computation, and Euler equation integration
//! for multisphere/clump particles.
//!
//! Each [`MultisphereBody`] represents one rigid body composed of multiple sub-spheres.
//! Bodies are stored in [`MultisphereBodyStore`] (one resource, not per-atom data).
//! Sub-sphere atoms reference their body via `body_id` in [`ClumpAtom`].

use std::f64::consts::PI;

use rand::Rng;

use super::{quat_rotate, ClumpSphereConfig};

// ── MultisphereBody ─────────────────────────────────────────────────────────

/// One rigid body composed of multiple overlapping spheres.
///
/// Contains full rigid body state (COM, quaternion, omega) and precomputed
/// inertia tensor in principal frame. Forces/torques are accumulated each
/// step from sub-sphere contacts, then used in Euler equation integration.
pub struct MultisphereBody {
    /// Unique body ID (matches `ClumpAtom::body_id` for sub-spheres).
    pub id: u32,
    /// Center-of-mass position in space frame.
    pub com_pos: [f64; 3],
    /// Center-of-mass velocity in space frame.
    pub com_vel: [f64; 3],
    /// Orientation quaternion [w, x, y, z] (body → space rotation).
    pub quaternion: [f64; 4],
    /// Angular velocity in space frame (rad/s).
    pub omega: [f64; 3],
    /// Angular momentum in space frame.
    pub angmom: [f64; 3],
    /// Principal moments of inertia [Ix, Iy, Iz] in body frame.
    pub principal_moments: [f64; 3],
    /// Rotation from body frame to principal frame as quaternion [w, x, y, z].
    pub principal_axes: [f64; 4],
    /// Total mass of the rigid body.
    pub total_mass: f64,
    /// 1 / total_mass.
    pub inv_mass: f64,
    /// Accumulated force in space frame (zeroed each step).
    pub force: [f64; 3],
    /// Accumulated torque about COM in space frame (zeroed each step).
    pub torque: [f64; 3],
    /// PBC image flags for COM.
    pub image: [i32; 3],
    /// Body-frame offsets for each sub-sphere (fixed at creation).
    pub body_offsets: Vec<[f64; 3]>,
    /// Radius of each sub-sphere (fixed at creation).
    pub sub_sphere_radii: Vec<f64>,
    /// Global atom tags for each sub-sphere (for matching after reorder).
    pub sub_sphere_tags: Vec<u32>,
}

impl MultisphereBody {
    /// Zero accumulated force and torque.
    pub fn zero_accumulators(&mut self) {
        self.force = [0.0; 3];
        self.torque = [0.0; 3];
    }

    /// Number of sub-spheres in this body.
    pub fn num_spheres(&self) -> usize {
        self.body_offsets.len()
    }
}

/// Resource holding all multisphere rigid bodies.
///
/// Uses a flat array (`map`) indexed by body ID for O(1) lookups,
/// similar to LIGGGHTS's `mapArray_`.
#[derive(Default)]
pub struct MultisphereBodyStore {
    pub bodies: Vec<MultisphereBody>,
    /// Flat lookup: `map[body_id] = index` into `bodies`, or `usize::MAX` if absent.
    map: Vec<usize>,
}

impl MultisphereBodyStore {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            map: Vec::new(),
        }
    }

    /// Rebuild the flat ID → index map. Call after adding/removing bodies.
    pub fn generate_map(&mut self) {
        let max_id = self.bodies.iter().map(|b| b.id as usize).max().unwrap_or(0);
        self.map.clear();
        self.map.resize(max_id + 1, usize::MAX);
        for (idx, body) in self.bodies.iter().enumerate() {
            self.map[body.id as usize] = idx;
        }
    }

    /// O(1) lookup: body ID → index into `bodies`. Returns `None` if not found.
    #[inline]
    pub fn map(&self, id: u32) -> Option<usize> {
        let id = id as usize;
        if id < self.map.len() {
            let idx = self.map[id];
            if idx != usize::MAX { Some(idx) } else { None }
        } else {
            None
        }
    }

    /// Find body by ID. Returns index into `bodies`.
    pub fn find_by_id(&self, id: u32) -> Option<usize> {
        self.map(id)
    }
}

// ── Quaternion utilities (local) ────────────────────────────────────────────

/// Conjugate (inverse for unit quaternions) q = [w, x, y, z] → [w, -x, -y, -z].
#[inline]
pub fn quat_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Hamilton product of two quaternions.
#[inline]
pub fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// Normalize a quaternion to unit length.
#[inline]
pub fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm > 1e-30 {
        let inv = 1.0 / norm;
        [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

/// Rotate a 3-vector by the inverse of quaternion q (space → body frame).
#[inline]
pub fn quat_rotate_inv(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    quat_rotate(quat_conj(q), v)
}

// ── Angular momentum integration helpers (LIGGGHTS-style) ──────────────────

/// Build rotation matrix columns from quaternion q = [w, x, y, z].
/// Returns (ex, ey, ez) where each is a column of the rotation matrix R.
#[inline]
fn quat_to_rotation_columns(q: [f64; 4]) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let w = q[0];
    let x = q[1];
    let y = q[2];
    let z = q[3];
    let ex = [
        w * w + x * x - y * y - z * z,
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ];
    let ey = [
        2.0 * (x * y - w * z),
        w * w - x * x + y * y - z * z,
        2.0 * (y * z + w * x),
    ];
    let ez = [
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        w * w - x * x - y * y + z * z,
    ];
    (ex, ey, ez)
}

/// Convert angular momentum (space frame) to angular velocity (space frame)
/// using the combined orientation quaternion (principal→space) and principal
/// moments of inertia.
///
/// The `quat_principal_to_space` should be `body.quaternion * body.principal_axes`
/// (composed before calling), so the rotation columns map from principal frame
/// to space frame.
///
/// Matches LIGGGHTS `MathExtra::angmom_to_omega`.
#[inline]
pub fn angmom_to_omega(
    angmom: [f64; 3],
    quat_principal_to_space: [f64; 4],
    moments: [f64; 3],
) -> [f64; 3] {
    let (ex, ey, ez) = quat_to_rotation_columns(quat_principal_to_space);

    // Project angular momentum into principal frame: L_principal = R^T * L_space
    let lpx = ex[0] * angmom[0] + ex[1] * angmom[1] + ex[2] * angmom[2];
    let lpy = ey[0] * angmom[0] + ey[1] * angmom[1] + ey[2] * angmom[2];
    let lpz = ez[0] * angmom[0] + ez[1] * angmom[1] + ez[2] * angmom[2];

    // omega_principal = L_principal / I (guard against zero inertia)
    let wpx = if moments[0] > 1e-30 { lpx / moments[0] } else { 0.0 };
    let wpy = if moments[1] > 1e-30 { lpy / moments[1] } else { 0.0 };
    let wpz = if moments[2] > 1e-30 { lpz / moments[2] } else { 0.0 };

    // Rotate back to space frame: omega = R * omega_principal
    [
        ex[0] * wpx + ey[0] * wpy + ez[0] * wpz,
        ex[1] * wpx + ey[1] * wpy + ez[1] * wpz,
        ex[2] * wpx + ey[2] * wpy + ez[2] * wpz,
    ]
}

/// Quaternion kinematic equation: dq/dt = 0.5 * omega_quat * q.
///
/// Returns the time derivative components (not scaled by dt).
/// Matches LIGGGHTS `MathExtra::vecquat`.
#[inline]
fn vecquat(w: [f64; 3], q: [f64; 4]) -> [f64; 4] {
    [
        -w[0] * q[1] - w[1] * q[2] - w[2] * q[3],
         q[0] * w[0] + w[1] * q[3] - w[2] * q[2],
         q[0] * w[1] + w[2] * q[1] - w[0] * q[3],
         q[0] * w[2] + w[0] * q[2] - w[1] * q[1],
    ]
}

/// Richardson extrapolation for quaternion integration.
///
/// Uses two estimates (full step and two half-steps) for second-order accuracy.
/// Matches LIGGGHTS `FixMultisphere::richardson`.
///
/// `q` is the body quaternion (body→space). `principal_axes` is the body→principal
/// rotation; composed internally for `angmom_to_omega` calls.
fn richardson(
    q: &mut [f64; 4],
    angmom: [f64; 3],
    omega: [f64; 3],
    moments: [f64; 3],
    dtq: f64,
    principal_axes: [f64; 4],
) {
    let half_dtq = 0.5 * dtq;

    // Full Euler step
    let dq = vecquat(omega, *q);
    let q_full = quat_normalize([
        q[0] + dtq * dq[0],
        q[1] + dtq * dq[1],
        q[2] + dtq * dq[2],
        q[3] + dtq * dq[3],
    ]);

    // First half step
    let mut q_half = quat_normalize([
        q[0] + half_dtq * dq[0],
        q[1] + half_dtq * dq[1],
        q[2] + half_dtq * dq[2],
        q[3] + half_dtq * dq[3],
    ]);

    // Recompute omega at the half-step orientation
    let q_half_ps = quat_mul(q_half, principal_axes);
    let omega_half = angmom_to_omega(angmom, q_half_ps, moments);

    // Second half step
    let dq2 = vecquat(omega_half, q_half);
    q_half = quat_normalize([
        q_half[0] + half_dtq * dq2[0],
        q_half[1] + half_dtq * dq2[1],
        q_half[2] + half_dtq * dq2[2],
        q_half[3] + half_dtq * dq2[3],
    ]);

    // Richardson extrapolation: 2*q_half - q_full
    *q = quat_normalize([
        2.0 * q_half[0] - q_full[0],
        2.0 * q_half[1] - q_full[1],
        2.0 * q_half[2] - q_full[2],
        2.0 * q_half[3] - q_full[3],
    ]);
}

// ── Inertia tensor computation ──────────────────────────────────────────────

/// Compute full 3×3 inertia tensor analytically (parallel axis theorem).
///
/// Assumes non-overlapping spheres. Each sphere contributes:
/// - Diagonal: I_sphere + m*(d² - d_a²) for component a
/// - Off-diagonal: -m * d_a * d_b
///
/// Returns `(total_mass, tensor_3x3)` where tensor is symmetric.
pub fn compute_inertia_tensor_analytical(
    spheres: &[ClumpSphereConfig],
    density: f64,
) -> (f64, [[f64; 3]; 3]) {
    let mut total_mass = 0.0;
    let mut tensor = [[0.0_f64; 3]; 3];

    for s in spheres {
        let r = s.radius;
        let m = density * (4.0 / 3.0) * PI * r * r * r;
        let i_sphere = 0.4 * m * r * r; // 2/5 m r²
        let d = s.offset;

        let d_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];

        // Parallel axis theorem: I_ab += (I_sphere + m*d²)*delta_ab - m*d_a*d_b
        for a in 0..3 {
            for b in 0..3 {
                let delta_ab = if a == b { 1.0 } else { 0.0 };
                tensor[a][b] += (i_sphere + m * d_sq) * delta_ab - m * d[a] * d[b];
            }
        }

        total_mass += m;
    }

    (total_mass, tensor)
}

/// Compute inertia tensor via Monte Carlo sampling (handles overlapping spheres).
///
/// Samples random points within the bounding box; points inside any sphere
/// contribute to the inertia tensor with uniform density.
///
/// Returns `(total_mass, tensor_3x3)`.
pub fn compute_inertia_tensor_montecarlo(
    spheres: &[ClumpSphereConfig],
    density: f64,
    n_samples: usize,
) -> (f64, [[f64; 3]; 3]) {
    // Compute bounding box
    let mut bb_min = [f64::MAX; 3];
    let mut bb_max = [f64::MIN; 3];
    for s in spheres {
        for d in 0..3 {
            bb_min[d] = bb_min[d].min(s.offset[d] - s.radius);
            bb_max[d] = bb_max[d].max(s.offset[d] + s.radius);
        }
    }

    let bb_size = [
        bb_max[0] - bb_min[0],
        bb_max[1] - bb_min[1],
        bb_max[2] - bb_min[2],
    ];
    let bb_volume = bb_size[0] * bb_size[1] * bb_size[2];

    let mut rng = rand::rng();
    let mut hits = 0u64;
    let mut tensor = [[0.0_f64; 3]; 3];

    for _ in 0..n_samples {
        let p = [
            bb_min[0] + rng.random::<f64>() * bb_size[0],
            bb_min[1] + rng.random::<f64>() * bb_size[1],
            bb_min[2] + rng.random::<f64>() * bb_size[2],
        ];

        // Check if point is inside any sphere
        let inside = spheres.iter().any(|s| {
            let dx = p[0] - s.offset[0];
            let dy = p[1] - s.offset[1];
            let dz = p[2] - s.offset[2];
            dx * dx + dy * dy + dz * dz <= s.radius * s.radius
        });

        if inside {
            hits += 1;
            // r² * delta_ab - r_a * r_b
            for a in 0..3 {
                for b in 0..3 {
                    let r_sq = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
                    let delta_ab = if a == b { 1.0 } else { 0.0 };
                    tensor[a][b] += r_sq * delta_ab - p[a] * p[b];
                }
            }
        }
    }

    // Scale: each sample point represents a volume element dV = bb_volume / n_samples
    // I = density * integral(r²δ - rr) dV ≈ density * (bb_volume / n_samples) * sum
    let dv = bb_volume / n_samples as f64;
    let total_volume = dv * hits as f64;
    let total_mass = density * total_volume;

    for a in 0..3 {
        for b in 0..3 {
            tensor[a][b] *= density * dv;
        }
    }

    (total_mass, tensor)
}

/// Check if any pair of spheres overlaps.
pub fn has_overlap(spheres: &[ClumpSphereConfig]) -> bool {
    for i in 0..spheres.len() {
        for j in (i + 1)..spheres.len() {
            let dx = spheres[i].offset[0] - spheres[j].offset[0];
            let dy = spheres[i].offset[1] - spheres[j].offset[1];
            let dz = spheres[i].offset[2] - spheres[j].offset[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < spheres[i].radius + spheres[j].radius {
                return true;
            }
        }
    }
    false
}

// ── Jacobi eigendecomposition for 3×3 symmetric matrices ────────────────────

/// Jacobi eigendecomposition for a 3×3 symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvectors[col] is the
/// eigenvector for eigenvalues[col]. Eigenvectors form a rotation matrix.
pub fn jacobi_eigendecomposition(mat: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = mat;
    // Eigenvector matrix (starts as identity)
    let mut v = [[0.0_f64; 3]; 3];
    v[0][0] = 1.0;
    v[1][1] = 1.0;
    v[2][2] = 1.0;

    for _ in 0..50 {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-15 {
            break;
        }

        // Compute rotation angle
        let theta = if (a[p][p] - a[q][q]).abs() < 1e-30 {
            PI / 4.0
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Givens rotation: A' = G^T A G
        let mut new_a = a;

        // Update rows/cols p and q
        for k in 0..3 {
            if k != p && k != q {
                new_a[k][p] = c * a[k][p] + s * a[k][q];
                new_a[p][k] = new_a[k][p];
                new_a[k][q] = -s * a[k][p] + c * a[k][q];
                new_a[q][k] = new_a[k][q];
            }
        }
        new_a[p][p] = c * c * a[p][p] + 2.0 * s * c * a[p][q] + s * s * a[q][q];
        new_a[q][q] = s * s * a[p][p] - 2.0 * s * c * a[p][q] + c * c * a[q][q];
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;

        a = new_a;

        // Update eigenvectors: V' = V * G
        for k in 0..3 {
            let vkp = v[k][p];
            let vkq = v[k][q];
            v[k][p] = c * vkp + s * vkq;
            v[k][q] = -s * vkp + c * vkq;
        }
    }

    let eigenvalues = [a[0][0], a[1][1], a[2][2]];
    (eigenvalues, v)
}

/// Convert a 3×3 rotation matrix (column-major eigenvectors) to a quaternion.
///
/// The matrix columns are the eigenvectors from Jacobi decomposition.
pub fn rotation_matrix_to_quaternion(m: [[f64; 3]; 3]) -> [f64; 4] {
    // m[row][col] — each column is an eigenvector
    let trace = m[0][0] + m[1][1] + m[2][2];

    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0; // s = 4*w
        let w = 0.25 * s;
        let x = (m[2][1] - m[1][2]) / s;
        let y = (m[0][2] - m[2][0]) / s;
        let z = (m[1][0] - m[0][1]) / s;
        quat_normalize([w, x, y, z])
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        let w = (m[2][1] - m[1][2]) / s;
        let x = 0.25 * s;
        let y = (m[0][1] + m[1][0]) / s;
        let z = (m[0][2] + m[2][0]) / s;
        quat_normalize([w, x, y, z])
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        let w = (m[0][2] - m[2][0]) / s;
        let x = (m[0][1] + m[1][0]) / s;
        let y = 0.25 * s;
        let z = (m[1][2] + m[2][1]) / s;
        quat_normalize([w, x, y, z])
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        let w = (m[1][0] - m[0][1]) / s;
        let x = (m[0][2] + m[2][0]) / s;
        let y = (m[1][2] + m[2][1]) / s;
        let z = 0.25 * s;
        quat_normalize([w, x, y, z])
    }
}

/// Diagonalize an inertia tensor: returns (principal_moments, principal_axes_quaternion).
pub fn diagonalize_inertia(tensor: [[f64; 3]; 3]) -> ([f64; 3], [f64; 4]) {
    let (eigenvalues, eigenvectors) = jacobi_eigendecomposition(tensor);
    let q = rotation_matrix_to_quaternion(eigenvectors);
    (eigenvalues, q)
}

// ── Euler equation integration ──────────────────────────────────────────────

/// Initial half-step integration for a rigid body (Velocity Verlet).
///
/// Uses angular momentum as state variable with Richardson-extrapolated
/// quaternion integration, matching LIGGGHTS `fix_multisphere`.
///
/// 1. Translational half-kick velocity + full drift position
/// 2. Half-kick angular momentum from torque
/// 3. Derive omega from angular momentum
/// 4. Richardson-extrapolated quaternion update
/// 5. Recompute omega with updated quaternion
pub fn integrate_body_initial(body: &mut MultisphereBody, dt: f64) {
    let half_dt = 0.5 * dt;

    // --- Translational: half-kick velocity, then drift position ---
    for d in 0..3 {
        body.com_vel[d] += half_dt * body.force[d] * body.inv_mass;
    }
    for d in 0..3 {
        body.com_pos[d] += dt * body.com_vel[d];
    }

    // --- Rotational: angular momentum integration ---
    // Half-kick angular momentum
    for d in 0..3 {
        body.angmom[d] += half_dt * body.torque[d];
    }

    // Combined quaternion: principal frame → space frame
    let q_ps = quat_mul(body.quaternion, body.principal_axes);

    // Derive omega from angular momentum
    body.omega = angmom_to_omega(body.angmom, q_ps, body.principal_moments);

    // Richardson-extrapolated quaternion update (dtq = dt/2 per LIGGGHTS convention)
    richardson(
        &mut body.quaternion,
        body.angmom,
        body.omega,
        body.principal_moments,
        half_dt,
        body.principal_axes,
    );

    // Recompute omega with the updated quaternion
    let q_ps = quat_mul(body.quaternion, body.principal_axes);
    body.omega = angmom_to_omega(body.angmom, q_ps, body.principal_moments);
}

/// Final half-step integration for a rigid body (Velocity Verlet).
///
/// Uses angular momentum as state variable, matching LIGGGHTS.
/// No quaternion update in the final step.
///
/// 1. Translational half-kick velocity
/// 2. Half-kick angular momentum from torque
/// 3. Derive omega from angular momentum
pub fn integrate_body_final(body: &mut MultisphereBody, dt: f64) {
    let half_dt = 0.5 * dt;

    // --- Translational half-kick ---
    for d in 0..3 {
        body.com_vel[d] += half_dt * body.force[d] * body.inv_mass;
    }

    // --- Rotational: angular momentum half-kick ---
    for d in 0..3 {
        body.angmom[d] += half_dt * body.torque[d];
    }

    // Derive omega from angular momentum
    let q_ps = quat_mul(body.quaternion, body.principal_axes);
    body.omega = angmom_to_omega(body.angmom, q_ps, body.principal_moments);
}

// ── Pack / Unpack for future MPI ────────────────────────────────────────────

impl MultisphereBody {
    /// Pack full body state into a flat f64 buffer.
    pub fn pack(&self, buf: &mut Vec<f64>) {
        buf.push(self.id as f64);
        buf.extend_from_slice(&self.com_pos);
        buf.extend_from_slice(&self.com_vel);
        buf.push(self.quaternion[0]);
        buf.push(self.quaternion[1]);
        buf.push(self.quaternion[2]);
        buf.push(self.quaternion[3]);
        buf.extend_from_slice(&self.omega);
        buf.extend_from_slice(&self.angmom);
        buf.extend_from_slice(&self.principal_moments);
        buf.push(self.principal_axes[0]);
        buf.push(self.principal_axes[1]);
        buf.push(self.principal_axes[2]);
        buf.push(self.principal_axes[3]);
        buf.push(self.total_mass);
        buf.push(self.inv_mass);
        buf.extend_from_slice(&self.force);
        buf.extend_from_slice(&self.torque);
        buf.push(self.image[0] as f64);
        buf.push(self.image[1] as f64);
        buf.push(self.image[2] as f64);
        // Variable-length sub-sphere data
        let n = self.body_offsets.len();
        buf.push(n as f64);
        for i in 0..n {
            buf.extend_from_slice(&self.body_offsets[i]);
            buf.push(self.sub_sphere_radii[i]);
            buf.push(self.sub_sphere_tags[i] as f64);
        }
    }

    /// Unpack a body from a flat f64 buffer. Returns number of f64s consumed.
    pub fn unpack(buf: &[f64]) -> (MultisphereBody, usize) {
        let mut p = 0;
        let id = buf[p] as u32;
        p += 1;
        let com_pos = [buf[p], buf[p + 1], buf[p + 2]];
        p += 3;
        let com_vel = [buf[p], buf[p + 1], buf[p + 2]];
        p += 3;
        let quaternion = [buf[p], buf[p + 1], buf[p + 2], buf[p + 3]];
        p += 4;
        let omega = [buf[p], buf[p + 1], buf[p + 2]];
        p += 3;
        let angmom = [buf[p], buf[p + 1], buf[p + 2]];
        p += 3;
        let principal_moments = [buf[p], buf[p + 1], buf[p + 2]];
        p += 3;
        let principal_axes = [buf[p], buf[p + 1], buf[p + 2], buf[p + 3]];
        p += 4;
        let total_mass = buf[p];
        p += 1;
        let inv_mass = buf[p];
        p += 1;
        let force = [buf[p], buf[p + 1], buf[p + 2]];
        p += 3;
        let torque = [buf[p], buf[p + 1], buf[p + 2]];
        p += 3;
        let image = [buf[p] as i32, buf[p + 1] as i32, buf[p + 2] as i32];
        p += 3;
        let n = buf[p] as usize;
        p += 1;
        let mut body_offsets = Vec::with_capacity(n);
        let mut sub_sphere_radii = Vec::with_capacity(n);
        let mut sub_sphere_tags = Vec::with_capacity(n);
        for _ in 0..n {
            body_offsets.push([buf[p], buf[p + 1], buf[p + 2]]);
            p += 3;
            sub_sphere_radii.push(buf[p]);
            p += 1;
            sub_sphere_tags.push(buf[p] as u32);
            p += 1;
        }

        (
            MultisphereBody {
                id,
                com_pos,
                com_vel,
                quaternion,
                omega,
                angmom,
                principal_moments,
                principal_axes,
                total_mass,
                inv_mass,
                force,
                torque,
                image,
                body_offsets,
                sub_sphere_radii,
                sub_sphere_tags,
            },
            p,
        )
    }

    /// Pack forward-comm data (COM pos/vel/quaternion/omega).
    pub fn pack_forward(&self, buf: &mut Vec<f64>) {
        buf.extend_from_slice(&self.com_pos);
        buf.extend_from_slice(&self.com_vel);
        buf.push(self.quaternion[0]);
        buf.push(self.quaternion[1]);
        buf.push(self.quaternion[2]);
        buf.push(self.quaternion[3]);
        buf.extend_from_slice(&self.omega);
    }

    /// Pack reverse-comm data (force/torque for accumulation).
    pub fn pack_reverse(&self, buf: &mut Vec<f64>) {
        buf.extend_from_slice(&self.force);
        buf.extend_from_slice(&self.torque);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_diagonal_matrix() {
        // Already diagonal — eigenvalues should match
        let mat = [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]];
        let (vals, _vecs) = jacobi_eigendecomposition(mat);
        let mut sorted = vals;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-10);
        assert!((sorted[1] - 2.0).abs() < 1e-10);
        assert!((sorted[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn jacobi_known_symmetric() {
        // Symmetric matrix with known eigenvalues
        let mat = [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let (vals, vecs) = jacobi_eigendecomposition(mat);

        // Verify A*v = lambda*v for each eigenpair
        for col in 0..3 {
            let v = [vecs[0][col], vecs[1][col], vecs[2][col]];
            let av = [
                mat[0][0] * v[0] + mat[0][1] * v[1] + mat[0][2] * v[2],
                mat[1][0] * v[0] + mat[1][1] * v[1] + mat[1][2] * v[2],
                mat[2][0] * v[0] + mat[2][1] * v[1] + mat[2][2] * v[2],
            ];
            for d in 0..3 {
                assert!(
                    (av[d] - vals[col] * v[d]).abs() < 1e-10,
                    "Eigenpair {} component {} failed: Av={}, lv={}",
                    col,
                    d,
                    av[d],
                    vals[col] * v[d]
                );
            }
        }
    }

    #[test]
    fn montecarlo_single_sphere() {
        let spheres = vec![ClumpSphereConfig {
            offset: [0.0, 0.0, 0.0],
            radius: 1.0,
        }];
        let density = 1.0;
        let (mass, tensor) = compute_inertia_tensor_montecarlo(&spheres, density, 500_000);

        let expected_mass = density * (4.0 / 3.0) * PI;
        let expected_i = 0.4 * expected_mass; // 2/5 m r² with r=1

        // MC has statistical error — allow 5%
        assert!(
            (mass - expected_mass).abs() / expected_mass < 0.05,
            "mass: got {}, expected {}",
            mass,
            expected_mass
        );

        // Diagonal elements should all be ≈ expected_i
        for d in 0..3 {
            assert!(
                (tensor[d][d] - expected_i).abs() / expected_i < 0.05,
                "I[{}][{}]: got {}, expected {}",
                d,
                d,
                tensor[d][d],
                expected_i
            );
        }

        // Off-diagonal should be near zero
        for a in 0..3 {
            for b in 0..3 {
                if a != b {
                    assert!(
                        tensor[a][b].abs() / expected_i < 0.05,
                        "I[{}][{}] = {} should be near zero",
                        a,
                        b,
                        tensor[a][b]
                    );
                }
            }
        }
    }

    #[test]
    fn analytical_vs_montecarlo_nonoverlapping_dimer() {
        let spheres = vec![
            ClumpSphereConfig {
                offset: [-2.0, 0.0, 0.0],
                radius: 1.0,
            },
            ClumpSphereConfig {
                offset: [2.0, 0.0, 0.0],
                radius: 1.0,
            },
        ];
        let density = 1.0;

        let (_mass_a, tensor_a) = compute_inertia_tensor_analytical(&spheres, density);
        let (_mass_mc, tensor_mc) = compute_inertia_tensor_montecarlo(&spheres, density, 500_000);

        // Compare diagonal elements (within MC noise ~5%)
        for d in 0..3 {
            let rel_err = (tensor_a[d][d] - tensor_mc[d][d]).abs()
                / tensor_a[d][d].abs().max(1e-30);
            assert!(
                rel_err < 0.10,
                "I[{}][{}]: analytical={}, MC={}, rel_err={}",
                d,
                d,
                tensor_a[d][d],
                tensor_mc[d][d],
                rel_err
            );
        }
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let body = MultisphereBody {
            id: 42,
            com_pos: [1.0, 2.0, 3.0],
            com_vel: [0.1, 0.2, 0.3],
            quaternion: [1.0, 0.0, 0.0, 0.0],
            omega: [10.0, 20.0, 30.0],
            angmom: [15.0, 50.0, 105.0],
            principal_moments: [1.5, 2.5, 3.5],
            principal_axes: [1.0, 0.0, 0.0, 0.0],
            total_mass: 5.0,
            inv_mass: 0.2,
            force: [0.0; 3],
            torque: [0.0; 3],
            image: [1, -2, 3],
            body_offsets: vec![[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            sub_sphere_radii: vec![0.3, 0.4],
            sub_sphere_tags: vec![100, 101],
        };

        let mut buf = Vec::new();
        body.pack(&mut buf);

        let (unpacked, consumed) = MultisphereBody::unpack(&buf);
        assert_eq!(consumed, buf.len());
        assert_eq!(unpacked.id, 42);
        assert_eq!(unpacked.com_pos, [1.0, 2.0, 3.0]);
        assert_eq!(unpacked.image, [1, -2, 3]);
        assert_eq!(unpacked.body_offsets.len(), 2);
        assert_eq!(unpacked.sub_sphere_tags, vec![100, 101]);
    }

    #[test]
    fn euler_torque_free_spinning() {
        // A symmetric body spinning about z with no torque should conserve energy
        let mut body = MultisphereBody {
            id: 1,
            com_pos: [0.0; 3],
            com_vel: [0.0; 3],
            quaternion: [1.0, 0.0, 0.0, 0.0],
            omega: [0.0, 0.0, 100.0],
            angmom: [0.0, 0.0, 200.0], // L = I * omega = 2.0 * 100.0
            principal_moments: [1.0, 1.0, 2.0],
            principal_axes: [1.0, 0.0, 0.0, 0.0],
            total_mass: 1.0,
            inv_mass: 1.0,
            force: [0.0; 3],
            torque: [0.0; 3],
            image: [0; 3],
            body_offsets: vec![],
            sub_sphere_radii: vec![],
            sub_sphere_tags: vec![],
        };

        let dt = 1e-4;
        let initial_ke = 0.5 * body.principal_moments[2] * body.omega[2] * body.omega[2];

        for _ in 0..10000 {
            integrate_body_initial(&mut body, dt);
            integrate_body_final(&mut body, dt);
        }

        // Rotational KE: 0.5 * I . omega^2 (in principal frame)
        let q = body.quaternion;
        let pa = body.principal_axes;
        let omega_body = quat_rotate(quat_conj(q), body.omega);
        let omega_p = quat_rotate(quat_conj(pa), omega_body);
        let final_ke = 0.5
            * (body.principal_moments[0] * omega_p[0] * omega_p[0]
                + body.principal_moments[1] * omega_p[1] * omega_p[1]
                + body.principal_moments[2] * omega_p[2] * omega_p[2]);

        let rel_err = (final_ke - initial_ke).abs() / initial_ke;
        assert!(
            rel_err < 0.01,
            "Energy not conserved: initial={}, final={}, rel_err={}",
            initial_ke,
            final_ke,
            rel_err
        );
    }

    #[test]
    fn pure_torque_angular_acceleration() {
        // Apply torque about x-axis on a body with known moments
        let ix = 2.0;
        let mut body = MultisphereBody {
            id: 1,
            com_pos: [0.0; 3],
            com_vel: [0.0; 3],
            quaternion: [1.0, 0.0, 0.0, 0.0],
            omega: [0.0; 3],
            angmom: [0.0; 3],
            principal_moments: [ix, 3.0, 4.0],
            principal_axes: [1.0, 0.0, 0.0, 0.0],
            total_mass: 1.0,
            inv_mass: 1.0,
            force: [0.0; 3],
            torque: [10.0, 0.0, 0.0], // torque about x
            image: [0; 3],
            body_offsets: vec![],
            sub_sphere_radii: vec![],
            sub_sphere_tags: vec![],
        };

        let dt = 1e-5;
        // After one full step: omega_x ≈ dt * tau_x / Ix
        integrate_body_initial(&mut body, dt);
        body.torque = [10.0, 0.0, 0.0]; // Re-apply for final step
        integrate_body_final(&mut body, dt);

        let expected_omega_x = dt * 10.0 / ix;
        assert!(
            (body.omega[0] - expected_omega_x).abs() < 1e-10,
            "omega_x: got {}, expected {}",
            body.omega[0],
            expected_omega_x
        );
    }

    #[test]
    fn has_overlap_detection() {
        // Non-overlapping
        let spheres_no = vec![
            ClumpSphereConfig { offset: [-2.0, 0.0, 0.0], radius: 0.5 },
            ClumpSphereConfig { offset: [2.0, 0.0, 0.0], radius: 0.5 },
        ];
        assert!(!has_overlap(&spheres_no));

        // Overlapping
        let spheres_yes = vec![
            ClumpSphereConfig { offset: [-0.3, 0.0, 0.0], radius: 1.0 },
            ClumpSphereConfig { offset: [0.3, 0.0, 0.0], radius: 1.0 },
        ];
        assert!(has_overlap(&spheres_yes));
    }
}

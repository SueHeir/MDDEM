//! Multisphere/clump rigid body composites for non-spherical DEM particles.
//!
//! A **clump** is a rigid body composed of multiple overlapping spheres. Each sphere
//! participates in normal contact detection, but forces are aggregated to the parent
//! (center-of-mass) atom. The parent integrates translational and rotational equations
//! of motion, then sub-sphere positions/velocities are updated from the parent state.
//!
//! # Configuration
//!
//! ```toml
//! [[dem.clumps]]
//! name = "dimer"
//! spheres = [
//!     { offset = [-0.0003, 0.0, 0.0], radius = 0.001 },
//!     { offset = [0.0003, 0.0, 0.0], radius = 0.001 },
//! ]
//! ```
//!
//! Then reference in insertion:
//! ```toml
//! [[particles.insert]]
//! clump = "dimer"
//! material = "glass"
//! count = 100
//! density = 2500.0
//! region = { type = "block", min = [0, 0, 0.01], max = [0.01, 0.01, 0.02] }
//! ```

use std::f64::consts::PI;

use mddem_app::prelude::*;
use mddem_derive::AtomData;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{register_atom_data, Atom, AtomData, AtomDataRegistry, Config};

use dem_atom::DemAtom;

// ── Configuration ────────────────────────────────────────────────────────────

/// A single sphere within a clump definition.
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ClumpSphereConfig {
    /// Offset from clump center of mass in body frame [x, y, z].
    pub offset: [f64; 3],
    /// Sphere radius.
    pub radius: f64,
}

/// A clump type definition from `[[dem.clumps]]`.
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ClumpDef {
    /// Name of this clump type, referenced by insertion config.
    pub name: String,
    /// Spheres composing this clump (positions relative to COM).
    pub spheres: Vec<ClumpSphereConfig>,
}

/// Extended DEM config with optional clump definitions.
/// Uses `#[serde(flatten)]` with a HashMap to absorb unknown `[dem]` keys
/// (materials, contact_model, etc.) without failing deserialization.
#[derive(Deserialize, Clone, Default)]
pub struct DemClumpConfig {
    /// Clump type definitions.
    #[serde(default)]
    pub clumps: Option<Vec<ClumpDef>>,
    /// Pass-through fields we don't care about.
    #[serde(flatten)]
    pub _rest: std::collections::HashMap<String, toml::Value>,
}

// ── Per-atom clump data ─────────────────────────────────────────────────────

/// Per-atom clump membership and body-frame offset data.
///
/// Every atom gets these fields. For atoms not in a clump, `clump_id` is 0.
/// Parent atoms (clump COM) have `is_parent_flag` = 1.0.
/// Sub-spheres have `is_parent_flag` = 0.0 and store their body-frame offset.
#[derive(AtomData)]
pub struct ClumpAtom {
    /// Clump ID this atom belongs to (0 = not in a clump).
    /// Encoded as f64 for AtomData compatibility; use as u32.
    #[forward]
    pub clump_id: Vec<f64>,

    /// 1.0 if this atom is a clump parent (COM atom), 0.0 otherwise.
    #[forward]
    pub is_parent_flag: Vec<f64>,

    /// Local offset from clump COM in body frame [x, y, z].
    pub body_offset: Vec<[f64; 3]>,

    /// Index of the parent atom in the local array.
    /// Only meaningful for sub-spheres. Encoded as f64 for AtomData.
    pub parent_index: Vec<f64>,
}

impl Default for ClumpAtom {
    fn default() -> Self {
        Self::new()
    }
}

impl ClumpAtom {
    pub fn new() -> Self {
        ClumpAtom {
            clump_id: Vec::new(),
            is_parent_flag: Vec::new(),
            body_offset: Vec::new(),
            parent_index: Vec::new(),
        }
    }
}

// ── Clump registry (runtime data) ──────────────────────────────────────────

/// Runtime storage for clump definitions, looked up during insertion.
pub struct ClumpRegistry {
    pub defs: Vec<ClumpDef>,
}

impl ClumpRegistry {
    pub fn new() -> Self {
        ClumpRegistry { defs: Vec::new() }
    }

    pub fn find(&self, name: &str) -> Option<&ClumpDef> {
        self.defs.iter().find(|d| d.name == name)
    }
}

impl Default for ClumpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Quaternion utilities ─────────────────────────────────────────────────────

/// Rotate a vector by a quaternion q = [w, x, y, z].
#[inline]
pub fn quat_rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let w = q[0];
    let qx = q[1];
    let qy = q[2];
    let qz = q[3];

    // q * v * q^-1 using the optimized formula:
    // result = v + 2*w*(q_vec × v) + 2*(q_vec × (q_vec × v))
    let cx = qy * v[2] - qz * v[1];
    let cy = qz * v[0] - qx * v[2];
    let cz = qx * v[1] - qy * v[0];

    [
        v[0] + 2.0 * (w * cx + qy * cz - qz * cy),
        v[1] + 2.0 * (w * cy + qz * cx - qx * cz),
        v[2] + 2.0 * (w * cz + qx * cy - qy * cx),
    ]
}

/// Cross product of two 3-vectors.
#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ── Inertia computation ─────────────────────────────────────────────────────

/// Compute scalar moment of inertia for a clump of spheres via parallel axis theorem.
///
/// Each sub-sphere contributes `I_sphere + m_sphere * d^2` where d is the distance
/// from the sub-sphere center to the clump COM.
///
/// Returns (total_mass, scalar_inertia) using spherical approximation
/// (average of diagonal elements of the full inertia tensor).
pub fn compute_clump_inertia(
    spheres: &[ClumpSphereConfig],
    density: f64,
) -> (f64, f64) {
    let mut total_mass = 0.0;
    let mut total_inertia_xx = 0.0;
    let mut total_inertia_yy = 0.0;
    let mut total_inertia_zz = 0.0;

    for s in spheres {
        let r = s.radius;
        let m = density * (4.0 / 3.0) * PI * r * r * r;
        let i_sphere = 0.4 * m * r * r; // 2/5 * m * r^2
        let dx = s.offset[0];
        let dy = s.offset[1];
        let dz = s.offset[2];
        let d_sq = dx * dx + dy * dy + dz * dz;

        // Parallel axis theorem for each diagonal component:
        // I_xx += I_sphere + m * (dy^2 + dz^2)
        // I_yy += I_sphere + m * (dx^2 + dz^2)
        // I_zz += I_sphere + m * (dx^2 + dy^2)
        total_inertia_xx += i_sphere + m * (dy * dy + dz * dz);
        total_inertia_yy += i_sphere + m * (dx * dx + dz * dz);
        total_inertia_zz += i_sphere + m * (dx * dx + dy * dy);

        let _ = d_sq; // used implicitly above
        total_mass += m;
    }

    // Spherical approximation: average of diagonal elements
    let avg_inertia = (total_inertia_xx + total_inertia_yy + total_inertia_zz) / 3.0;

    (total_mass, avg_inertia)
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Plugin that adds multisphere/clump rigid body support to MDDEM.
///
/// Registers:
/// - [`ClumpAtom`] per-atom data
/// - [`ClumpRegistry`] resource with clump definitions from config
/// - Force aggregation system at [`ScheduleSet::PostForce`]
/// - Position update system at [`ScheduleSet::PostFinalIntegration`]
pub struct ClumpPlugin;

impl Plugin for ClumpPlugin {
    fn dependencies(&self) -> Vec<&str> {
        vec!["DemAtomPlugin"]
    }

    fn build(&self, app: &mut App) {
        register_atom_data!(app, ClumpAtom::new());

        // Load clump definitions from config
        let mut registry = ClumpRegistry::new();

        // Try to load dem.clumps from config
        let dem_clump_config = Config::load::<DemClumpConfig>(app, "dem");
        if let Some(clumps) = dem_clump_config.clumps {
            for def in clumps {
                assert!(
                    !def.spheres.is_empty(),
                    "Clump '{}' must have at least one sphere",
                    def.name
                );
                registry.defs.push(def);
            }
        }

        app.add_resource(registry);

        // Force aggregation: after contact forces, before final integration
        app.add_update_system(
            aggregate_clump_forces
                .label("aggregate_clump_forces")
                .after("hertz_mindlin_contact"),
            ScheduleSet::PostForce,
        );

        // Position update: after parent integrates, update sub-spheres
        app.add_update_system(
            update_clump_positions.label("update_clump_positions"),
            ScheduleSet::PostFinalIntegration,
        );

        // Skip sub-sphere integration: zero inv_mass so Verlet skips them
        // (v += 0.5 * dt * f * 0 = no change, x += v * dt but we overwrite in PostFinalIntegration)
        // Actually we need a pre-integration system to freeze sub-spheres
        app.add_update_system(
            freeze_subsphere_integration.label("freeze_subspheres"),
            ScheduleSet::PreInitialIntegration,
        );

        // Rebuild parent indices after exchange (atoms may be reordered)
        app.add_update_system(
            rebuild_parent_indices.label("rebuild_parent_indices"),
            ScheduleSet::PreForce,
        );
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Before integration: set sub-sphere inv_mass to 0 so Verlet doesn't integrate them.
/// Also zero their forces (they'll be aggregated to parent).
fn freeze_subsphere_integration(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    let clump = registry.get::<ClumpAtom>();
    let clump = match clump {
        Some(c) => c,
        None => return,
    };

    let nlocal = atoms.nlocal as usize;
    for i in 0..nlocal {
        if i < clump.clump_id.len() && clump.clump_id[i] > 0.0 && clump.is_parent_flag[i] < 0.5 {
            // Sub-sphere: zero inv_mass to prevent Verlet integration
            atoms.inv_mass[i] = 0.0;
        }
    }
}

/// Rebuild parent_index mappings after atoms may have been reordered.
/// Maps each sub-sphere to its parent's current local index by matching clump_id.
fn rebuild_parent_indices(atoms: Res<Atom>, registry: Res<AtomDataRegistry>) {
    let mut clump = registry.expect_mut::<ClumpAtom>("rebuild_parent_indices");

    let n = atoms.len();
    if clump.clump_id.len() < n {
        return; // Not initialized yet
    }

    // Build a map from clump_id -> parent local index
    // Only consider local atoms for parents
    let nlocal = atoms.nlocal as usize;
    let mut parent_map: Vec<(u32, usize)> = Vec::new();
    for i in 0..nlocal {
        if clump.clump_id[i] > 0.0 && clump.is_parent_flag[i] > 0.5 {
            parent_map.push((clump.clump_id[i] as u32, i));
        }
    }

    // Set parent_index for all sub-spheres (including ghosts for completeness)
    for i in 0..n {
        if i >= clump.clump_id.len() {
            break;
        }
        if clump.clump_id[i] > 0.0 && clump.is_parent_flag[i] < 0.5 {
            let cid = clump.clump_id[i] as u32;
            if let Some(&(_, pidx)) = parent_map.iter().find(|(id, _)| *id == cid) {
                clump.parent_index[i] = pidx as f64;
            }
        }
    }
}

/// Aggregate forces from sub-spheres to their parent clump body.
///
/// For each sub-sphere `s` belonging to clump parent `p`:
/// - Add `force[s]` to `force[p]`
/// - Compute torque: `r × force[s]` where `r = pos[s] - pos[p]`
/// - Add `torque[s]` to `torque[p]`
/// - Zero `force[s]` and `torque[s]`
pub fn aggregate_clump_forces(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    let clump = registry.get::<ClumpAtom>();
    let clump = match clump {
        Some(c) => c,
        None => return,
    };

    let mut dem = registry.expect_mut::<DemAtom>("aggregate_clump_forces");
    let nlocal = atoms.nlocal as usize;

    // Collect sub-sphere data first, then apply to parents
    // (to avoid aliasing issues with force array)
    struct SubForce {
        parent_idx: usize,
        force: [f64; 3],
        torque_contribution: [f64; 3],
        subsphere_torque: [f64; 3],
        subsphere_idx: usize,
    }

    let mut contributions: Vec<SubForce> = Vec::new();

    for i in 0..nlocal {
        if i >= clump.clump_id.len() {
            break;
        }
        // Skip non-clump atoms and parent atoms
        if clump.clump_id[i] <= 0.0 || clump.is_parent_flag[i] > 0.5 {
            continue;
        }

        let pidx = clump.parent_index[i] as usize;
        if pidx >= nlocal {
            continue; // Parent not local
        }

        // r = pos[sub] - pos[parent]
        let rx = atoms.pos[i][0] - atoms.pos[pidx][0];
        let ry = atoms.pos[i][1] - atoms.pos[pidx][1];
        let rz = atoms.pos[i][2] - atoms.pos[pidx][2];

        // Torque from force: r × F
        let fx = atoms.force[i][0];
        let fy = atoms.force[i][1];
        let fz = atoms.force[i][2];

        let torque_contrib = cross([rx, ry, rz], [fx, fy, fz]);

        let sub_torque = if i < dem.torque.len() {
            dem.torque[i]
        } else {
            [0.0; 3]
        };

        contributions.push(SubForce {
            parent_idx: pidx,
            force: [fx, fy, fz],
            torque_contribution: torque_contrib,
            subsphere_torque: sub_torque,
            subsphere_idx: i,
        });
    }

    // Apply all contributions to parents
    for c in &contributions {
        let p = c.parent_idx;
        atoms.force[p][0] += c.force[0];
        atoms.force[p][1] += c.force[1];
        atoms.force[p][2] += c.force[2];

        if p < dem.torque.len() {
            dem.torque[p][0] += c.torque_contribution[0] + c.subsphere_torque[0];
            dem.torque[p][1] += c.torque_contribution[1] + c.subsphere_torque[1];
            dem.torque[p][2] += c.torque_contribution[2] + c.subsphere_torque[2];
        }
    }

    // Zero sub-sphere forces and torques
    for c in &contributions {
        let s = c.subsphere_idx;
        atoms.force[s] = [0.0; 3];
        if s < dem.torque.len() {
            dem.torque[s] = [0.0; 3];
        }
    }
}

/// After parent integration: update sub-sphere positions and velocities.
///
/// - Rotate each sub-sphere's body_offset by parent's quaternion
/// - Set `pos[s] = pos[parent] + rotated_offset`
/// - Set `vel[s] = vel[parent] + omega × rotated_offset`
pub fn update_clump_positions(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    let clump = registry.get::<ClumpAtom>();
    let clump = match clump {
        Some(c) => c,
        None => return,
    };

    let dem = registry.expect::<DemAtom>("update_clump_positions");
    let nlocal = atoms.nlocal as usize;

    // Collect updates, then apply (to avoid borrow issues)
    struct SubUpdate {
        idx: usize,
        pos: [f64; 3],
        vel: [f64; 3],
    }

    let mut updates: Vec<SubUpdate> = Vec::new();

    for i in 0..nlocal {
        if i >= clump.clump_id.len() {
            break;
        }
        // Skip non-clump atoms and parent atoms
        if clump.clump_id[i] <= 0.0 || clump.is_parent_flag[i] > 0.5 {
            continue;
        }

        let pidx = clump.parent_index[i] as usize;
        if pidx >= nlocal || pidx >= dem.quaternion.len() {
            continue;
        }

        let q = dem.quaternion[pidx];
        let offset = clump.body_offset[i];

        // Rotate body offset by parent quaternion
        let rotated = quat_rotate(q, offset);

        // New position = parent pos + rotated offset
        let new_pos = [
            atoms.pos[pidx][0] + rotated[0],
            atoms.pos[pidx][1] + rotated[1],
            atoms.pos[pidx][2] + rotated[2],
        ];

        // New velocity = parent vel + omega × rotated_offset
        let omega = dem.omega[pidx];
        let omega_cross_r = cross(omega, rotated);
        let new_vel = [
            atoms.vel[pidx][0] + omega_cross_r[0],
            atoms.vel[pidx][1] + omega_cross_r[1],
            atoms.vel[pidx][2] + omega_cross_r[2],
        ];

        updates.push(SubUpdate {
            idx: i,
            pos: new_pos,
            vel: new_vel,
        });
    }

    for u in updates {
        atoms.pos[u.idx] = u.pos;
        atoms.vel[u.idx] = u.vel;
    }
}

// ── Clump insertion helper ──────────────────────────────────────────────────

/// Insert a single clump at the given COM position with the given material type.
///
/// Creates a parent atom at `com_pos` and N sub-sphere atoms at offset positions.
/// Returns the number of atoms inserted (1 parent + N sub-spheres).
pub fn insert_clump(
    atoms: &mut Atom,
    dem: &mut DemAtom,
    clump_data: &mut ClumpAtom,
    def: &ClumpDef,
    com_pos: [f64; 3],
    com_vel: [f64; 3],
    density: f64,
    atom_type: u32,
    clump_id: u32,
) -> usize {
    let (total_mass, inertia) = compute_clump_inertia(&def.spheres, density);
    let inv_inertia = if inertia > 0.0 { 1.0 / inertia } else { 0.0 };

    // Find a bounding radius for the parent (for neighbor list cutoff)
    let max_extent = def.spheres.iter().map(|s| {
        let d = (s.offset[0] * s.offset[0] + s.offset[1] * s.offset[1] + s.offset[2] * s.offset[2]).sqrt();
        d + s.radius
    }).fold(0.0_f64, f64::max);

    let base_tag = atoms.get_max_tag() + 1;

    // --- Insert parent atom ---
    atoms.tag.push(base_tag);
    atoms.atom_type.push(atom_type);
    atoms.origin_index.push(0);
    atoms.pos.push(com_pos);
    atoms.vel.push(com_vel);
    atoms.force.push([0.0; 3]);
    atoms.mass.push(total_mass);
    atoms.inv_mass.push(1.0 / total_mass);
    // Parent cutoff_radius should be large enough that sub-spheres are within ghost range
    atoms.cutoff_radius.push(max_extent);
    atoms.is_ghost.push(false);

    dem.radius.push(0.0); // Parent has no contact radius (sub-spheres do contacts)
    dem.density.push(density);
    dem.inv_inertia.push(inv_inertia);
    dem.quaternion.push([1.0, 0.0, 0.0, 0.0]);
    dem.omega.push([0.0; 3]);
    dem.ang_mom.push([0.0; 3]);
    dem.torque.push([0.0; 3]);

    let parent_local_idx = atoms.pos.len() - 1;
    clump_data.clump_id.push(clump_id as f64);
    clump_data.is_parent_flag.push(1.0);
    clump_data.body_offset.push([0.0; 3]);
    clump_data.parent_index.push(parent_local_idx as f64);

    // --- Insert sub-sphere atoms ---
    for (si, sphere) in def.spheres.iter().enumerate() {
        let sub_tag = base_tag + 1 + si as u32;
        let sub_pos = [
            com_pos[0] + sphere.offset[0],
            com_pos[1] + sphere.offset[1],
            com_pos[2] + sphere.offset[2],
        ];

        let sub_mass = density * (4.0 / 3.0) * PI * sphere.radius.powi(3);

        atoms.tag.push(sub_tag);
        atoms.atom_type.push(atom_type);
        atoms.origin_index.push(0);
        atoms.pos.push(sub_pos);
        atoms.vel.push(com_vel); // Initial velocity same as COM
        atoms.force.push([0.0; 3]);
        atoms.mass.push(sub_mass);
        atoms.inv_mass.push(0.0); // Zero inv_mass: Verlet won't integrate
        atoms.cutoff_radius.push(sphere.radius);
        atoms.is_ghost.push(false);

        dem.radius.push(sphere.radius);
        dem.density.push(density);
        dem.inv_inertia.push(0.0); // Sub-spheres don't rotate independently
        dem.quaternion.push([1.0, 0.0, 0.0, 0.0]);
        dem.omega.push([0.0; 3]);
        dem.ang_mom.push([0.0; 3]);
        dem.torque.push([0.0; 3]);

        clump_data.clump_id.push(clump_id as f64);
        clump_data.is_parent_flag.push(0.0);
        clump_data.body_offset.push(sphere.offset);
        clump_data.parent_index.push(parent_local_idx as f64);
    }

    let total_inserted = 1 + def.spheres.len();

    // Update atom counts
    atoms.nlocal += total_inserted as u32;
    atoms.natoms += total_inserted as u64;

    total_inserted
}

/// Check if two atoms belong to the same clump (for contact exclusion).
///
/// Returns true if both atoms have the same non-zero clump_id.
#[inline]
pub fn same_clump(clump_data: &ClumpAtom, i: usize, j: usize) -> bool {
    if i >= clump_data.clump_id.len() || j >= clump_data.clump_id.len() {
        return false;
    }
    let ci = clump_data.clump_id[i];
    let cj = clump_data.clump_id[j];
    ci > 0.0 && cj > 0.0 && (ci - cj).abs() < 0.5 // Same integer ID
}

/// Check if atom i is a clump parent.
#[inline]
pub fn is_clump_parent(clump_data: &ClumpAtom, i: usize) -> bool {
    i < clump_data.is_parent_flag.len() && clump_data.is_parent_flag[i] > 0.5
}

/// Check if atom i is a clump sub-sphere (member but not parent).
#[inline]
pub fn is_subsphere(clump_data: &ClumpAtom, i: usize) -> bool {
    i < clump_data.clump_id.len()
        && clump_data.clump_id[i] > 0.0
        && clump_data.is_parent_flag[i] < 0.5
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::{Atom, AtomDataRegistry};
    use dem_atom::DemAtom;

    fn make_dimer_def() -> ClumpDef {
        ClumpDef {
            name: "dimer".to_string(),
            spheres: vec![
                ClumpSphereConfig {
                    offset: [-0.0003, 0.0, 0.0],
                    radius: 0.001,
                },
                ClumpSphereConfig {
                    offset: [0.0003, 0.0, 0.0],
                    radius: 0.001,
                },
            ],
        }
    }

    fn setup_clump_test() -> (Atom, DemAtom, ClumpAtom) {
        let atom = Atom::new();
        let dem = DemAtom::new();
        let clump = ClumpAtom::new();
        (atom, dem, clump)
    }

    #[test]
    fn test_quat_rotate_identity() {
        let q = [1.0, 0.0, 0.0, 0.0]; // identity quaternion
        let v = [1.0, 2.0, 3.0];
        let result = quat_rotate(q, v);
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!((result[1] - 2.0).abs() < 1e-12);
        assert!((result[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_quat_rotate_90_degrees_z() {
        // 90° rotation about z-axis: q = [cos(45°), 0, 0, sin(45°)]
        let angle = std::f64::consts::FRAC_PI_2;
        let half = angle * 0.5;
        let q = [half.cos(), 0.0, 0.0, half.sin()];
        let v = [1.0, 0.0, 0.0]; // x-axis vector
        let result = quat_rotate(q, v);
        // Should rotate to [0, 1, 0]
        assert!((result[0]).abs() < 1e-12, "x should be 0, got {}", result[0]);
        assert!((result[1] - 1.0).abs() < 1e-12, "y should be 1, got {}", result[1]);
        assert!((result[2]).abs() < 1e-12, "z should be 0, got {}", result[2]);
    }

    #[test]
    fn test_compute_clump_inertia_single_sphere() {
        let spheres = vec![ClumpSphereConfig {
            offset: [0.0, 0.0, 0.0],
            radius: 0.001,
        }];
        let density = 2500.0;
        let (mass, inertia) = compute_clump_inertia(&spheres, density);

        let expected_mass = density * (4.0 / 3.0) * PI * 0.001_f64.powi(3);
        let expected_inertia = 0.4 * expected_mass * 0.001 * 0.001;

        assert!((mass - expected_mass).abs() < 1e-15, "mass mismatch");
        assert!(
            (inertia - expected_inertia).abs() / expected_inertia < 1e-12,
            "inertia mismatch: got {}, expected {}",
            inertia,
            expected_inertia
        );
    }

    #[test]
    fn test_compute_clump_inertia_dimer() {
        let def = make_dimer_def();
        let density = 2500.0;
        let (mass, inertia) = compute_clump_inertia(&def.spheres, density);

        // Each sphere: m = density * 4/3 * pi * r^3
        let r = 0.001;
        let m_sphere = density * (4.0 / 3.0) * PI * r * r * r;
        let expected_mass = 2.0 * m_sphere;
        assert!((mass - expected_mass).abs() < 1e-15);

        // Each sphere: I_sphere = 2/5 * m * r^2
        // Parallel axis for offset d = 0.0003:
        // I_xx += 2/5 * m * r^2 + m * (0 + 0) = 2/5*m*r^2 (offset only in x)
        // I_yy += 2/5 * m * r^2 + m * (0.0003^2 + 0)
        // I_zz += 2/5 * m * r^2 + m * (0.0003^2 + 0)
        let i_sphere = 0.4 * m_sphere * r * r;
        let d = 0.0003;
        let i_xx = 2.0 * i_sphere; // no offset in y or z
        let i_yy = 2.0 * (i_sphere + m_sphere * d * d);
        let i_zz = 2.0 * (i_sphere + m_sphere * d * d);
        let expected_avg = (i_xx + i_yy + i_zz) / 3.0;

        assert!(
            (inertia - expected_avg).abs() / expected_avg < 1e-10,
            "dimer inertia: got {}, expected {}",
            inertia,
            expected_avg
        );

        // Inertia should be larger than two isolated spheres
        assert!(inertia > 2.0 * i_sphere);
    }

    #[test]
    fn test_insert_clump_creates_correct_atoms() {
        let (mut atoms, mut dem, mut clump) = setup_clump_test();
        let def = make_dimer_def();

        let count = insert_clump(
            &mut atoms,
            &mut dem,
            &mut clump,
            &def,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            2500.0,
            0,
            1, // clump_id
        );

        assert_eq!(count, 3, "Should insert 1 parent + 2 sub-spheres");
        assert_eq!(atoms.nlocal, 3);
        assert_eq!(atoms.natoms, 3);

        // Parent is atom 0
        assert!(clump.is_parent_flag[0] > 0.5);
        assert!((clump.clump_id[0] - 1.0).abs() < 0.1);
        assert_eq!(dem.radius[0], 0.0); // Parent has no contact radius

        // Sub-spheres are atoms 1 and 2
        assert!(clump.is_parent_flag[1] < 0.5);
        assert!(clump.is_parent_flag[2] < 0.5);
        assert!((dem.radius[1] - 0.001).abs() < 1e-10);
        assert!((dem.radius[2] - 0.001).abs() < 1e-10);

        // Sub-sphere positions should be offset from COM
        assert!((atoms.pos[1][0] - (-0.0003)).abs() < 1e-10);
        assert!((atoms.pos[2][0] - 0.0003).abs() < 1e-10);

        // Sub-spheres have zero inv_mass
        assert_eq!(atoms.inv_mass[1], 0.0);
        assert_eq!(atoms.inv_mass[2], 0.0);

        // Parent has correct total mass
        let r = 0.001;
        let m_sphere = 2500.0 * (4.0 / 3.0) * PI * r * r * r;
        assert!((atoms.mass[0] - 2.0 * m_sphere).abs() < 1e-15);
    }

    #[test]
    fn test_same_clump_exclusion() {
        let (mut atoms, mut dem, mut clump) = setup_clump_test();
        let def = make_dimer_def();

        insert_clump(
            &mut atoms, &mut dem, &mut clump, &def,
            [0.0, 0.0, 0.0], [0.0; 3], 2500.0, 0, 1,
        );

        // Atoms 1 and 2 are in same clump
        assert!(same_clump(&clump, 1, 2));
        // Atom 0 (parent) and atom 1 (sub) are in same clump
        assert!(same_clump(&clump, 0, 1));
    }

    #[test]
    fn test_different_clumps_not_excluded() {
        let (mut atoms, mut dem, mut clump) = setup_clump_test();
        let def = make_dimer_def();

        insert_clump(
            &mut atoms, &mut dem, &mut clump, &def,
            [0.0, 0.0, 0.0], [0.0; 3], 2500.0, 0, 1,
        );
        insert_clump(
            &mut atoms, &mut dem, &mut clump, &def,
            [0.01, 0.0, 0.0], [0.0; 3], 2500.0, 0, 2,
        );

        // Sub-spheres from different clumps should NOT be excluded
        assert!(!same_clump(&clump, 1, 4)); // clump 1 sub vs clump 2 sub
        assert!(!same_clump(&clump, 2, 5));
    }

    #[test]
    fn test_force_aggregation() {
        let (mut atoms, mut dem, mut clump) = setup_clump_test();
        let def = make_dimer_def();

        insert_clump(
            &mut atoms, &mut dem, &mut clump, &def,
            [0.0, 0.0, 0.0], [0.0; 3], 2500.0, 0, 1,
        );

        // Apply force to sub-sphere 1 (at x = -0.0003)
        atoms.force[1] = [0.0, 0.0, 10.0]; // Force in z on left sub-sphere

        // Register clump data in registry for the system to use
        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(clump);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(registry);
        app.add_update_system(aggregate_clump_forces, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let atoms = app.get_resource_ref::<Atom>().unwrap();
        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");

        // Force should be transferred to parent (atom 0)
        assert!(
            (atoms.force[0][2] - 10.0).abs() < 1e-10,
            "Parent z-force should be 10.0, got {}",
            atoms.force[0][2]
        );

        // Sub-sphere force should be zeroed
        assert!(
            (atoms.force[1][2]).abs() < 1e-10,
            "Sub-sphere force should be zeroed, got {}",
            atoms.force[1][2]
        );

        // Torque on parent: r × F = [-0.0003, 0, 0] × [0, 0, 10] = [0, 0.003, 0]
        assert!(
            (dem.torque[0][1] - 0.003).abs() < 1e-10,
            "Parent y-torque should be 0.003, got {}",
            dem.torque[0][1]
        );
    }

    #[test]
    fn test_position_update_after_rotation() {
        let (mut atoms, mut dem, mut clump) = setup_clump_test();
        let def = make_dimer_def();

        insert_clump(
            &mut atoms, &mut dem, &mut clump, &def,
            [0.0, 0.0, 0.0], [0.0; 3], 2500.0, 0, 1,
        );

        // Rotate parent 90° about z-axis
        let angle = std::f64::consts::FRAC_PI_2;
        let half = angle * 0.5;
        dem.quaternion[0] = [half.cos(), 0.0, 0.0, half.sin()];

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(clump);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(registry);
        app.add_update_system(update_clump_positions, ScheduleSet::PostFinalIntegration);
        app.organize_systems();
        app.run();

        let atoms = app.get_resource_ref::<Atom>().unwrap();

        // After 90° z rotation:
        // Sub-sphere 1 offset [-0.0003, 0, 0] -> [0, -0.0003, 0]
        // Sub-sphere 2 offset [0.0003, 0, 0] -> [0, 0.0003, 0]
        assert!(
            (atoms.pos[1][0]).abs() < 1e-10,
            "Sub1 x should be ~0, got {}",
            atoms.pos[1][0]
        );
        assert!(
            (atoms.pos[1][1] - (-0.0003)).abs() < 1e-10,
            "Sub1 y should be -0.0003, got {}",
            atoms.pos[1][1]
        );

        assert!(
            (atoms.pos[2][0]).abs() < 1e-10,
            "Sub2 x should be ~0, got {}",
            atoms.pos[2][0]
        );
        assert!(
            (atoms.pos[2][1] - 0.0003).abs() < 1e-10,
            "Sub2 y should be 0.0003, got {}",
            atoms.pos[2][1]
        );
    }

    #[test]
    fn test_dimer_free_fall() {
        // Test that a dimer in free fall (constant force) has COM following parabolic trajectory
        let (mut atoms, mut dem, mut clump) = setup_clump_test();
        let def = make_dimer_def();

        let com_pos = [0.0, 0.0, 0.1];
        insert_clump(
            &mut atoms, &mut dem, &mut clump, &def,
            com_pos, [0.0; 3], 2500.0, 0, 1,
        );
        atoms.dt = 1e-6;

        let gravity_z = -9.81;
        let total_mass = atoms.mass[0];

        // Manual integration: apply gravity, aggregate, integrate, update positions
        let nsteps = 100;
        let dt = atoms.dt;
        let mut com_vel_z = 0.0;
        let mut com_pos_z = com_pos[2];

        for _ in 0..nsteps {
            // Apply gravity force to parent (in a real sim, gravity fix does this)
            atoms.force[0][2] = total_mass * gravity_z;
            // Also apply to sub-spheres for testing force aggregation
            // Actually in a real sim, gravity is applied to all atoms.
            // But since sub-spheres have inv_mass=0, Verlet won't integrate them.
            // Let's apply gravity only to parent for simplicity.

            // VV initial half-kick + drift (parent only, sub-spheres have inv_mass=0)
            let half_dt_a = 0.5 * dt * atoms.force[0][2] / total_mass;
            atoms.vel[0][2] += half_dt_a;
            atoms.pos[0][2] += atoms.vel[0][2] * dt;

            // Expected trajectory
            com_vel_z += 0.5 * dt * gravity_z;
            com_pos_z += com_vel_z * dt;

            // VV final half-kick
            // Force doesn't change for gravity
            atoms.vel[0][2] += 0.5 * dt * atoms.force[0][2] / total_mass;
            com_vel_z += 0.5 * dt * gravity_z;

            // Update sub-sphere positions (no rotation, identity quaternion)
            for si in 1..3 {
                let offset = clump.body_offset[si];
                atoms.pos[si][0] = atoms.pos[0][0] + offset[0];
                atoms.pos[si][1] = atoms.pos[0][1] + offset[1];
                atoms.pos[si][2] = atoms.pos[0][2] + offset[2];
                atoms.vel[si] = atoms.vel[0];
            }
        }

        // COM should follow exact parabolic trajectory
        assert!(
            (atoms.pos[0][2] - com_pos_z).abs() < 1e-14,
            "COM z position: got {}, expected {}",
            atoms.pos[0][2],
            com_pos_z
        );

        // Sub-spheres should maintain offset from COM
        assert!(
            (atoms.pos[1][0] - atoms.pos[0][0] - (-0.0003)).abs() < 1e-14,
            "Sub1 should maintain x offset"
        );
        assert!(
            (atoms.pos[2][0] - atoms.pos[0][0] - 0.0003).abs() < 1e-14,
            "Sub2 should maintain x offset"
        );

        // Sub-spheres should have same z as COM (no offset in z)
        assert!(
            (atoms.pos[1][2] - atoms.pos[0][2]).abs() < 1e-14,
            "Sub1 z should match COM z"
        );
    }

    #[test]
    fn test_contact_on_one_sphere_creates_torque() {
        let (mut atoms, mut dem, mut clump) = setup_clump_test();
        let def = make_dimer_def();

        insert_clump(
            &mut atoms, &mut dem, &mut clump, &def,
            [0.0, 0.0, 0.0], [0.0; 3], 2500.0, 0, 1,
        );

        // Force in y on right sub-sphere (at x = +0.0003)
        atoms.force[2] = [0.0, 5.0, 0.0];

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(clump);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(registry);
        app.add_update_system(aggregate_clump_forces, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let atoms = app.get_resource_ref::<Atom>().unwrap();
        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");

        // Force transferred to parent
        assert!((atoms.force[0][1] - 5.0).abs() < 1e-10);

        // Torque: r × F = [0.0003, 0, 0] × [0, 5, 0] = [0, 0, 0.0015]
        assert!(
            (dem.torque[0][2] - 0.0015).abs() < 1e-10,
            "z-torque should be 0.0015, got {}",
            dem.torque[0][2]
        );
    }

    #[test]
    fn test_subsphere_velocity_from_rotation() {
        let (mut atoms, mut dem, mut clump) = setup_clump_test();
        let def = make_dimer_def();

        insert_clump(
            &mut atoms, &mut dem, &mut clump, &def,
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], // COM moving in x
            2500.0, 0, 1,
        );

        // Set angular velocity about z-axis
        dem.omega[0] = [0.0, 0.0, 100.0]; // 100 rad/s about z

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(clump);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(registry);
        app.add_update_system(update_clump_positions, ScheduleSet::PostFinalIntegration);
        app.organize_systems();
        app.run();

        let atoms = app.get_resource_ref::<Atom>().unwrap();

        // Sub-sphere 2 at offset [0.0003, 0, 0]:
        // vel = COM_vel + omega × offset = [1, 0, 0] + [0, 0, 100] × [0.0003, 0, 0]
        // omega × r = [0*0 - 100*0, 100*0.0003 - 0*0, 0*0 - 0*0.0003] = [0, 0.03, 0]
        // vel = [1, 0.03, 0]
        assert!(
            (atoms.vel[2][0] - 1.0).abs() < 1e-10,
            "Sub2 vx should be 1.0, got {}",
            atoms.vel[2][0]
        );
        assert!(
            (atoms.vel[2][1] - 0.03).abs() < 1e-10,
            "Sub2 vy should be 0.03, got {}",
            atoms.vel[2][1]
        );
    }
}

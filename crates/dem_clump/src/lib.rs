//! Multisphere/clump rigid body composites for non-spherical DEM particles.
//!
//! A **clump** is a rigid body composed of multiple overlapping spheres. Each sphere
//! participates in normal contact detection, but forces are aggregated to its
//! [`MultisphereBody`]. The body integrates translational + rotational (Euler equation)
//! dynamics, then sub-sphere positions/velocities are derived from body state.
//!
//! # Architecture
//!
//! - **No phantom parent atom.** Body data lives in [`MultisphereBodyStore`], not in
//!   the atom arrays. Each sub-sphere atom references its body via `body_id` in
//!   [`ClumpAtom`].
//! - **Full inertia tensor.** Analytically via parallel axis theorem, or Monte Carlo
//!   for overlapping spheres. Diagonalized to principal moments + axes quaternion.
//! - **Euler equation integration.** Torque → principal frame, Euler α, half-kick ω.
//!
//! # Configuration
//!
//! Clump definitions and insertion live under the `[clump]` TOML section
//! (separate from `[dem]` which uses `deny_unknown_fields`).
//!
//! ```toml
//! [[clump.definitions]]
//! name = "dimer"
//! spheres = [
//!     { offset = [-0.0003, 0.0, 0.0], radius = 0.001 },
//!     { offset = [0.0003, 0.0, 0.0], radius = 0.001 },
//! ]
//!
//! [[clump.insert]]
//! definition = "dimer"
//! count = 100
//! density = 2500.0
//! material = "glass"
//! velocity = 0.5
//! region = { type = "block", min = [0.001, 0.001, 0.001], max = [0.019, 0.019, 0.019] }
//! ```

use std::collections::HashMap;
use std::f64::consts::PI;

use sim_app::prelude::*;
use mddem_derive::AtomData;
use sim_scheduler::prelude::*;
use serde::Deserialize;
use rand::Rng;

use mddem_core::{
    register_atom_data, Atom, AtomData, AtomDataRegistry, CommResource, Config, Domain, Region,
    RunState, ScheduleSet, ScheduleSetupSet,
};

#[cfg(feature = "mpi_backend")]
use mddem_core::CommTopology;

use dem_atom::DemAtom;

pub mod body;
pub use body::{
    MultisphereBody, MultisphereBodyStore,
    compute_inertia_tensor_analytical, compute_inertia_tensor_montecarlo,
    has_overlap, diagonalize_inertia, jacobi_eigendecomposition,
    rotation_matrix_to_quaternion,
};

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

/// A clump type definition from `[[clump.definitions]]`.
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ClumpDef {
    /// Name of this clump type, referenced by insertion config.
    pub name: String,
    /// Spheres composing this clump (positions relative to COM).
    pub spheres: Vec<ClumpSphereConfig>,
}

/// A clump insertion block from `[[clump.insert]]`.
#[derive(Deserialize, Clone, Debug)]
pub struct ClumpInsertConfig {
    /// Name of the clump definition to insert.
    pub definition: String,
    /// Number of clumps to insert.
    pub count: u32,
    /// Particle density (kg/m³).
    pub density: f64,
    /// Material name (must match a `[[dem.materials]]` entry).
    pub material: String,
    /// Random velocity magnitude (m/s). Each component is uniform in [-v, +v].
    #[serde(default)]
    pub velocity: Option<f64>,
    /// Insertion region. Defaults to domain bounds inset by effective clump radius.
    #[serde(default)]
    pub region: Option<Region>,
}

/// TOML `[clump]` — top-level clump configuration.
///
/// Separate from `[dem]` because `DemConfig` uses `deny_unknown_fields`.
#[derive(Deserialize, Clone, Default)]
pub struct ClumpTopConfig {
    #[serde(default)]
    pub definitions: Option<Vec<ClumpDef>>,
    #[serde(default)]
    pub insert: Option<Vec<ClumpInsertConfig>>,
}

// ── Per-atom clump data ─────────────────────────────────────────────────────

/// Per-atom clump membership and body-frame offset data.
///
/// Every atom gets these fields. For atoms not in a clump, `body_id` is 0.
/// Sub-spheres store their body-frame offset for position reconstruction.
#[derive(AtomData)]
pub struct ClumpAtom {
    /// Body ID this atom belongs to (0 = not in a clump).
    /// Encoded as f64 for AtomData compatibility; use as u32.
    #[forward]
    pub body_id: Vec<f64>,

    /// Local offset from body COM in body frame [x, y, z].
    #[forward]
    pub body_offset: Vec<[f64; 3]>,
}

impl Default for ClumpAtom {
    fn default() -> Self {
        Self::new()
    }
}

impl ClumpAtom {
    pub fn new() -> Self {
        ClumpAtom {
            body_id: Vec::new(),
            body_offset: Vec::new(),
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
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ── Legacy scalar inertia (kept for backward compatibility) ─────────────────

/// Compute scalar moment of inertia for a clump (spherical approximation).
/// Prefer [`compute_inertia_tensor_analytical`] for full tensor math.
pub fn compute_clump_inertia(
    spheres: &[ClumpSphereConfig],
    density: f64,
) -> (f64, f64) {
    let (mass, tensor) = compute_inertia_tensor_analytical(spheres, density);
    let avg = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;
    (mass, avg)
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Plugin that adds multisphere/clump rigid body support to MDDEM.
///
/// Registers:
/// - [`ClumpAtom`] per-atom data (body_id, body_offset)
/// - [`ClumpRegistry`] resource with clump definitions from config
/// - [`MultisphereBodyStore`] resource for rigid body state
/// - Body integration systems (Euler equations)
/// - Force aggregation + position update systems
pub struct ClumpPlugin;

impl Plugin for ClumpPlugin {
    fn dependencies(&self) -> Vec<std::any::TypeId> {
        sim_app::type_ids![dem_atom::DemAtomPlugin]
    }

    fn build(&self, app: &mut App) {
        register_atom_data!(app, ClumpAtom::new());

        let mut registry = ClumpRegistry::new();
        let clump_config = Config::load::<ClumpTopConfig>(app, "clump");
        if let Some(defs) = clump_config.definitions {
            for def in defs {
                assert!(
                    !def.spheres.is_empty(),
                    "Clump '{}' must have at least one sphere",
                    def.name
                );
                registry.defs.push(def);
            }
        }

        app.add_resource(registry);
        app.add_resource(MultisphereBodyStore::new());

        // Clump insertion from [[clump.insert]] config (runs after normal particle insertion)
        app.add_setup_system(
            clump_insert_atoms.label("clump_insert_atoms"),
            ScheduleSetupSet::Setup,
        );

        // Set minimum ghost cutoff for clumps BEFORE neighbor_setup computes bins.
        // neighbor_setup will use max(its_value, domain.ghost_cutoff).
        app.add_setup_system(
            extend_ghost_cutoff_for_clumps
                .label("extend_ghost_cutoff_for_clumps"),
            ScheduleSetupSet::Setup,
        );

        // Before any exchange, snap sub-sphere positions to body COM so they
        // always migrate to the same rank as their body. Must run before
        // exchange_bodies so all bodies are still local for the lookup.
        app.add_update_system(
            snap_subspheres_to_body_com
                .label("snap_subspheres_to_body_com")
                .before("exchange_bodies")
                .before("exchange"),
            ScheduleSet::Exchange,
        );

        // Body exchange: migrate bodies whose COM left the local subdomain.
        app.add_update_system(
            exchange_bodies
                .label("exchange_bodies")
                .before("exchange"),
            ScheduleSet::Exchange,
        );

        // After atom exchange, restore sub-sphere positions from body state
        // (undo the snap-to-COM done before exchange).
        app.add_update_system(
            restore_subsphere_positions
                .label("restore_subsphere_positions")
                .after("exchange"),
            ScheduleSet::Exchange,
        );

        // Body initial integration (half-kick + drift + quaternion update)
        app.add_update_system(
            integrate_bodies_initial.label("integrate_bodies_initial"),
            ScheduleSet::InitialIntegration,
        );

        // PBC for body COM
        app.add_update_system(
            pbc_multisphere_bodies.label("pbc_multisphere_bodies"),
            ScheduleSet::PostInitialIntegration,
        );

        // Force aggregation: sub-sphere forces → body force/torque
        // Must run after reverse_send_force so ghost sub-sphere forces are
        // accumulated back to their owning atoms before aggregation to bodies.
        app.add_update_system(
            aggregate_clump_forces
                .label("aggregate_clump_forces")
                .after("hertz_mindlin_contact")
                .after("reverse_send_force"),
            ScheduleSet::PostForce,
        );

        // Body final integration (half-kick after new forces)
        app.add_update_system(
            integrate_bodies_final.label("integrate_bodies_final"),
            ScheduleSet::FinalIntegration,
        );

        // Update sub-sphere positions before exchange so atoms migrate to the
        // same rank as their body (prevents orphan sub-spheres whose forces
        // would have no local body to aggregate to).
        app.add_update_system(
            update_clump_positions
                .label("update_clump_positions_pre_exchange")
                .after("pbc_multisphere_bodies"),
            ScheduleSet::PostInitialIntegration,
        );

        // Derive sub-sphere pos/vel from body state (end of step)
        app.add_update_system(
            update_clump_positions.label("update_clump_positions"),
            ScheduleSet::PostFinalIntegration,
        );

        // Lost atom detection (lightweight, every 1000 steps)
        app.add_update_system(
            check_lost_clump_atoms
                .label("check_lost_clump_atoms")
                .after("update_clump_positions"),
            ScheduleSet::PostFinalIntegration,
        );
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Extend `Domain::ghost_cutoff` by the maximum clump bounding radius.
///
/// Ensures all sub-spheres of bodies near subdomain boundaries are visible as
/// ghosts on the body-owning rank. Mirrors LIGGGHTS's `extend_cut_ghost()`.
fn extend_ghost_cutoff_for_clumps(
    clump_registry: Res<ClumpRegistry>,
    mut domain: ResMut<Domain>,
    comm: Res<CommResource>,
) {
    let mut max_r_bound: f64 = 0.0;
    for def in &clump_registry.defs {
        for sphere in &def.spheres {
            let r = (sphere.offset[0].powi(2) + sphere.offset[1].powi(2) + sphere.offset[2].powi(2)).sqrt()
                + sphere.radius;
            max_r_bound = max_r_bound.max(r);
        }
    }
    if max_r_bound > 0.0 {
        let extension = 2.0 * max_r_bound;
        domain.ghost_cutoff += extension;
        if comm.rank() == 0 {
            println!(
                "ClumpPlugin: extended ghost_cutoff by {:.6} (2 * R_bound={:.6}) → {:.6}",
                extension, max_r_bound, domain.ghost_cutoff
            );
        }
    }
}

/// Snap sub-sphere positions to their body COM before atom exchange.
///
/// This ensures sub-spheres always migrate to the same rank as their body.
/// Without this, a sub-sphere near a subdomain boundary could exchange to a
/// different rank than its body, causing orphaned forces.
fn snap_subspheres_to_body_com(
    mut atoms: ResMut<Atom>,
    bodies: Res<MultisphereBodyStore>,
    registry: Res<AtomDataRegistry>,
) {
    let clump = match registry.get::<ClumpAtom>() {
        Some(c) => c,
        None => return,
    };

    let nlocal = atoms.nlocal as usize;
    for i in 0..nlocal {
        if i >= clump.body_id.len() {
            break;
        }
        let bid = clump.body_id[i] as u32;
        if bid == 0 {
            continue;
        }
        if let Some(body_idx) = bodies.map(bid) {
            atoms.pos[i] = bodies.bodies[body_idx].com_pos;
        }
    }
}

/// Restore sub-sphere positions from body state after atom exchange.
///
/// Undoes the snap-to-COM and sets correct offset positions, velocities, and angular velocities.
/// Also regenerates the body ID→index map after body exchange.
fn restore_subsphere_positions(
    mut atoms: ResMut<Atom>,
    mut bodies: ResMut<MultisphereBodyStore>,
    registry: Res<AtomDataRegistry>,
) {
    bodies.generate_map();

    let clump = match registry.get::<ClumpAtom>() {
        Some(c) => c,
        None => return,
    };
    let mut dem = registry.expect_mut::<DemAtom>("restore_subsphere_positions");

    let nlocal = atoms.nlocal as usize;
    for i in 0..nlocal {
        if i >= clump.body_id.len() {
            break;
        }
        let bid = clump.body_id[i] as u32;
        if bid == 0 {
            continue;
        }
        if let Some(body_idx) = bodies.map(bid) {
            let body = &bodies.bodies[body_idx];
            let rotated = quat_rotate(body.quaternion, clump.body_offset[i]);
            atoms.pos[i] = [
                body.com_pos[0] + rotated[0],
                body.com_pos[1] + rotated[1],
                body.com_pos[2] + rotated[2],
            ];
            let omega_cross_r = cross(body.omega, rotated);
            atoms.vel[i] = [
                body.com_vel[0] + omega_cross_r[0],
                body.com_vel[1] + omega_cross_r[1],
                body.com_vel[2] + omega_cross_r[2],
            ];
            dem.omega[i] = body.omega;
        }
    }
}

/// Initial half-step: integrate all rigid bodies (Euler equations).
fn integrate_bodies_initial(atoms: Res<Atom>, mut bodies: ResMut<MultisphereBodyStore>) {
    let dt = atoms.dt;
    for body in &mut bodies.bodies {
        body::integrate_body_initial(body, dt);
    }
}

/// Final half-step: integrate all rigid bodies after new forces.
fn integrate_bodies_final(atoms: Res<Atom>, mut bodies: ResMut<MultisphereBodyStore>) {
    let dt = atoms.dt;
    for body in &mut bodies.bodies {
        body::integrate_body_final(body, dt);
    }
}

/// Wrap body COM through periodic boundaries and update body image flags.
fn pbc_multisphere_bodies(mut bodies: ResMut<MultisphereBodyStore>, domain: Res<Domain>) {
    for body in &mut bodies.bodies {
        for d in 0..3 {
            if domain.is_periodic(d) {
                let low = domain.boundaries_low[d];
                let size = domain.size[d];
                let high = low + size;
                if body.com_pos[d] < low {
                    body.com_pos[d] += size;
                    body.image[d] -= 1;
                } else if body.com_pos[d] >= high {
                    body.com_pos[d] -= size;
                    body.image[d] += 1;
                }
            }
        }
    }
}

/// Exchange bodies between processors when their COM leaves the local subdomain.
///
/// Mirrors the atom exchange pattern in `comm.rs`: for each dimension, scan bodies
/// whose COM is outside `[sub_domain_low, sub_domain_high)`, pack them, send to the
/// neighbor processor, receive incoming bodies, and rebuild the ID→index map.
#[cfg(feature = "mpi_backend")]
fn exchange_bodies(
    comm: Res<CommResource>,
    topo: Res<CommTopology>,
    mut bodies: ResMut<MultisphereBodyStore>,
    domain: Res<Domain>,
) {
    let decomp = comm.processor_decomposition();

    let mut lo_buf: Vec<f64> = Vec::new();
    let mut hi_buf: Vec<f64> = Vec::new();

    for dim in 0..3usize {
        if decomp[dim] == 1 {
            continue;
        }

        let lo_proc = topo.swap_directions[0][dim];
        let hi_proc = topo.swap_directions[1][dim];

        lo_buf.clear();
        hi_buf.clear();
        let mut lo_count = 0u32;
        let mut hi_count = 0u32;

        // Scan bodies in reverse, pack those with COM outside subdomain
        for i in (0..bodies.bodies.len()).rev() {
            let pos = bodies.bodies[i].com_pos[dim];
            if pos < domain.sub_domain_low[dim] {
                lo_count += 1;
                bodies.bodies[i].pack(&mut lo_buf);
                bodies.bodies.swap_remove(i);
            } else if pos >= domain.sub_domain_high[dim] {
                hi_count += 1;
                bodies.bodies[i].pack(&mut hi_buf);
                bodies.bodies.swap_remove(i);
            }
        }

        lo_buf.push(lo_count as f64);
        hi_buf.push(hi_count as f64);

        // Send lo, receive from hi
        if lo_proc != -1 && hi_proc != -1 {
            let msg = comm.sendrecv_f64(lo_proc, &lo_buf, hi_proc);
            unpack_bodies_from_msg(&msg, &mut bodies.bodies);
        } else if lo_proc != -1 {
            comm.send_f64(lo_proc, &lo_buf);
        } else if hi_proc != -1 {
            let msg = comm.recv_f64(hi_proc);
            unpack_bodies_from_msg(&msg, &mut bodies.bodies);
        }

        // Send hi, receive from lo
        if hi_proc != -1 && lo_proc != -1 {
            let msg = comm.sendrecv_f64(hi_proc, &hi_buf, lo_proc);
            unpack_bodies_from_msg(&msg, &mut bodies.bodies);
        } else if hi_proc != -1 {
            comm.send_f64(hi_proc, &hi_buf);
        } else if lo_proc != -1 {
            let msg = comm.recv_f64(lo_proc);
            unpack_bodies_from_msg(&msg, &mut bodies.bodies);
        }
    }

    bodies.generate_map();
}

/// Unpack bodies from a received message buffer.
#[cfg(feature = "mpi_backend")]
fn unpack_bodies_from_msg(msg: &[f64], bodies: &mut Vec<MultisphereBody>) {
    let count = msg[msg.len() - 1] as usize;
    let data = &msg[..msg.len() - 1];
    let mut pos = 0;
    for _ in 0..count {
        let (body, consumed) = MultisphereBody::unpack(&data[pos..]);
        bodies.push(body);
        pos += consumed;
    }
}

/// No-op body exchange for single-process builds.
#[cfg(not(feature = "mpi_backend"))]
fn exchange_bodies() {}

/// Aggregate forces from sub-sphere atoms onto their parent body.
///
/// For each sub-sphere with `body_id > 0`:
/// - Accumulate force onto body
/// - Compute torque: `r × F` where `r` is the rotated body offset
/// - Accumulate sub-sphere contact torque onto body
/// - Zero sub-sphere force and torque
pub fn aggregate_clump_forces(
    mut atoms: ResMut<Atom>,
    mut bodies: ResMut<MultisphereBodyStore>,
    registry: Res<AtomDataRegistry>,
) {
    let clump = registry.get::<ClumpAtom>();
    let clump = match clump {
        Some(c) => c,
        None => return,
    };

    let mut dem = registry.expect_mut::<DemAtom>("aggregate_clump_forces");

    // Zero body accumulators
    for body in &mut bodies.bodies {
        body.zero_accumulators();
    }

    let nlocal = atoms.nlocal as usize;

    // Collect contributions to avoid borrow conflicts
    struct Contrib {
        body_idx: usize,
        force: [f64; 3],
        torque: [f64; 3],
        atom_idx: usize,
    }

    let mut contribs = Vec::new();

    for i in 0..nlocal {
        if i >= clump.body_id.len() {
            break;
        }
        let bid = clump.body_id[i] as u32;
        if bid == 0 {
            continue;
        }

        let body_idx = match bodies.map(bid) {
            Some(idx) => idx,
            None => continue,
        };

        let body = &bodies.bodies[body_idx];

        // r = rotated body_offset (current space-frame displacement)
        let rotated = quat_rotate(body.quaternion, clump.body_offset[i]);

        let f = atoms.force[i];
        let torque_from_force = cross(rotated, f);

        let sub_torque = if i < dem.torque.len() {
            dem.torque[i]
        } else {
            [0.0; 3]
        };

        contribs.push(Contrib {
            body_idx,
            force: f,
            torque: [
                torque_from_force[0] + sub_torque[0],
                torque_from_force[1] + sub_torque[1],
                torque_from_force[2] + sub_torque[2],
            ],
            atom_idx: i,
        });
    }

    // Apply contributions to bodies
    for c in &contribs {
        let body = &mut bodies.bodies[c.body_idx];
        for d in 0..3 {
            body.force[d] += c.force[d];
            body.torque[d] += c.torque[d];
        }
    }

    // Zero sub-sphere forces and torques
    for c in &contribs {
        atoms.force[c.atom_idx] = [0.0; 3];
        if c.atom_idx < dem.torque.len() {
            dem.torque[c.atom_idx] = [0.0; 3];
        }
    }
}

/// Derive sub-sphere positions, velocities, and angular velocities from body state.
///
/// For each sub-sphere: `pos = COM + q * body_offset`, `vel = COM_vel + omega × (q * offset)`,
/// `omega = body.omega` (rigid body constraint — all sub-spheres share the body angular velocity).
pub fn update_clump_positions(
    mut atoms: ResMut<Atom>,
    bodies: Res<MultisphereBodyStore>,
    registry: Res<AtomDataRegistry>,
) {
    let clump = registry.get::<ClumpAtom>();
    let clump = match clump {
        Some(c) => c,
        None => return,
    };

    let nlocal = atoms.nlocal as usize;

    struct SubUpdate {
        idx: usize,
        pos: [f64; 3],
        vel: [f64; 3],
        omega: [f64; 3],
    }

    let mut updates: Vec<SubUpdate> = Vec::new();

    for i in 0..nlocal {
        if i >= clump.body_id.len() {
            break;
        }
        let bid = clump.body_id[i] as u32;
        if bid == 0 {
            continue;
        }

        let body_idx = match bodies.map(bid) {
            Some(idx) => idx,
            None => continue,
        };

        let body = &bodies.bodies[body_idx];
        let rotated = quat_rotate(body.quaternion, clump.body_offset[i]);

        let new_pos = [
            body.com_pos[0] + rotated[0],
            body.com_pos[1] + rotated[1],
            body.com_pos[2] + rotated[2],
        ];

        let omega_cross_r = cross(body.omega, rotated);
        let new_vel = [
            body.com_vel[0] + omega_cross_r[0],
            body.com_vel[1] + omega_cross_r[1],
            body.com_vel[2] + omega_cross_r[2],
        ];

        updates.push(SubUpdate {
            idx: i,
            pos: new_pos,
            vel: new_vel,
            omega: body.omega,
        });
    }

    let mut dem = registry.expect_mut::<DemAtom>("update_clump_positions");
    for u in updates {
        atoms.pos[u.idx] = u.pos;
        atoms.vel[u.idx] = u.vel;
        dem.omega[u.idx] = u.omega;
    }
}

/// Diagnostic: check that each body has the expected number of local sub-sphere atoms.
///
/// Runs every 1000 steps. Warns on mismatch but does not delete atoms.
fn check_lost_clump_atoms(
    atoms: Res<Atom>,
    bodies: Res<MultisphereBodyStore>,
    registry: Res<AtomDataRegistry>,
    comm: Res<CommResource>,
    run_state: Res<RunState>,
) {
    if run_state.total_cycle % 1000 != 0 {
        return;
    }

    let clump = match registry.get::<ClumpAtom>() {
        Some(c) => c,
        None => return,
    };

    let nlocal = atoms.nlocal as usize;
    let mut counts: HashMap<u32, usize> = HashMap::new();

    for i in 0..nlocal {
        if i >= clump.body_id.len() {
            break;
        }
        let bid = clump.body_id[i] as u32;
        if bid > 0 {
            *counts.entry(bid).or_default() += 1;
        }
    }

    for body in &bodies.bodies {
        let expected = body.sub_sphere_tags.len();
        let actual = counts.get(&body.id).copied().unwrap_or(0);
        if actual != expected {
            eprintln!(
                "WARNING: Body {} has {}/{} atoms on rank {}",
                body.id, actual, expected, comm.rank()
            );
        }
    }
}

// ── Clump insertion from config ──────────────────────────────────────────────

/// Setup system: insert clumps from `[[clump.insert]]` config blocks.
///
/// For each insertion block, looks up the named clump definition from the
/// [`ClumpRegistry`], then inserts `count` clumps at random non-overlapping
/// positions within the specified region.
fn clump_insert_atoms(
    comm: Res<CommResource>,
    domain: Res<Domain>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    clump_registry: Res<ClumpRegistry>,
    mut body_store: ResMut<MultisphereBodyStore>,
    clump_config: Res<ClumpTopConfig>,
    material_table: Res<dem_atom::MaterialTable>,
) {
    let inserts = match clump_config.insert {
        Some(ref v) => v,
        None => return,
    };

    if comm.rank() != 0 {
        return;
    }

    let mut dem_data = registry.expect_mut::<DemAtom>("clump_insert_atoms");
    let mut clump_data = registry.expect_mut::<ClumpAtom>("clump_insert_atoms");
    let mut rng = rand::rng();

    for insert in inserts {
        let def = clump_registry.find(&insert.definition).unwrap_or_else(|| {
            panic!(
                "Clump definition '{}' not found. Available: {:?}",
                insert.definition,
                clump_registry.defs.iter().map(|d| &d.name).collect::<Vec<_>>()
            );
        });

        // Resolve material index
        let mat_idx = material_table
            .names
            .iter()
            .position(|n| n == &insert.material)
            .unwrap_or_else(|| {
                panic!(
                    "Material '{}' not found in [[dem.materials]]",
                    insert.material
                );
            }) as u32;

        // Compute effective radius for overlap checks (max sub-sphere extent from COM)
        let eff_radius = def
            .spheres
            .iter()
            .map(|s| {
                let d = (s.offset[0].powi(2) + s.offset[1].powi(2) + s.offset[2].powi(2)).sqrt();
                d + s.radius
            })
            .fold(0.0_f64, f64::max);

        // Determine insertion region
        let region = insert.region.clone().unwrap_or_else(|| Region::Block {
            min: [
                domain.boundaries_low[0] + eff_radius,
                domain.boundaries_low[1] + eff_radius,
                domain.boundaries_low[2] + eff_radius,
            ],
            max: [
                domain.boundaries_high[0] - eff_radius,
                domain.boundaries_high[1] - eff_radius,
                domain.boundaries_high[2] - eff_radius,
            ],
        });

        println!(
            "ClumpInsert: inserting {} '{}' clumps (eff_r={:.4}mm, rho={}, mat='{}')",
            insert.count,
            insert.definition,
            eff_radius * 1000.0,
            insert.density,
            insert.material,
        );

        // Track inserted COM positions for overlap avoidance
        let mut com_positions: Vec<[f64; 3]> = Vec::new();
        let mut inserted = 0u32;
        let mut attempts = 0u64;
        let max_attempts = insert.count as u64 * 1_000_000;
        let mut next_clump_id = body_store.bodies.len() as u32 + 1;

        while inserted < insert.count && attempts < max_attempts {
            attempts += 1;
            let pos = region.random_point_inside(&mut rng);

            // Check overlap with existing clump COMs
            let min_sep = 2.0 * eff_radius * 1.05; // 5% margin
            let mut overlaps = false;

            // Check against existing atoms
            for i in 0..atoms.len() {
                let dx = pos[0] - atoms.pos[i][0];
                let dy = pos[1] - atoms.pos[i][1];
                let dz = pos[2] - atoms.pos[i][2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let min_d = eff_radius + atoms.cutoff_radius[i];
                if dist_sq < min_d * min_d {
                    overlaps = true;
                    break;
                }
            }

            // Check against already-inserted clump COMs in this batch
            if !overlaps {
                for existing in &com_positions {
                    let dx = pos[0] - existing[0];
                    let dy = pos[1] - existing[1];
                    let dz = pos[2] - existing[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    if dist_sq < min_sep * min_sep {
                        overlaps = true;
                        break;
                    }
                }
            }

            if overlaps {
                continue;
            }

            // Generate velocity
            let vel = if let Some(v_mag) = insert.velocity {
                [
                    rng.random_range(-v_mag..v_mag),
                    rng.random_range(-v_mag..v_mag),
                    rng.random_range(-v_mag..v_mag),
                ]
            } else {
                [0.0; 3]
            };

            insert_clump(
                &mut atoms,
                &mut dem_data,
                &mut clump_data,
                &mut body_store,
                def,
                pos,
                vel,
                insert.density,
                mat_idx,
                next_clump_id,
            );

            com_positions.push(pos);
            next_clump_id += 1;
            inserted += 1;
        }

        if inserted < insert.count {
            eprintln!(
                "WARNING: Could only insert {}/{} clumps after {} attempts.",
                inserted, insert.count, max_attempts
            );
        }
    }
}

// ── Clump insertion helper ──────────────────────────────────────────────────

/// Insert a single clump at the given COM position.
///
/// Creates N sub-sphere atoms and one [`MultisphereBody`] entry.
/// No parent atom is created — the body resource holds COM state.
///
/// Returns the number of atoms inserted (N sub-spheres).
pub fn insert_clump(
    atoms: &mut Atom,
    dem: &mut DemAtom,
    clump_data: &mut ClumpAtom,
    body_store: &mut MultisphereBodyStore,
    def: &ClumpDef,
    com_pos: [f64; 3],
    com_vel: [f64; 3],
    density: f64,
    atom_type: u32,
    clump_id: u32,
) -> usize {
    // Compute inertia tensor (auto-detect overlap)
    let (total_mass, tensor) = if has_overlap(&def.spheres) {
        compute_inertia_tensor_montecarlo(&def.spheres, density, 100_000)
    } else {
        compute_inertia_tensor_analytical(&def.spheres, density)
    };

    let (principal_moments, principal_axes) = diagonalize_inertia(tensor);

    let base_tag = atoms.get_max_tag() + 1;

    // Create MultisphereBody
    let mut body_offsets = Vec::with_capacity(def.spheres.len());
    let mut sub_sphere_radii = Vec::with_capacity(def.spheres.len());
    let mut sub_sphere_tags = Vec::with_capacity(def.spheres.len());

    for (si, sphere) in def.spheres.iter().enumerate() {
        let sub_tag = base_tag + si as u32;
        body_offsets.push(sphere.offset);
        sub_sphere_radii.push(sphere.radius);
        sub_sphere_tags.push(sub_tag);
    }

    let body = MultisphereBody {
        id: clump_id,
        com_pos,
        com_vel,
        quaternion: [1.0, 0.0, 0.0, 0.0],
        omega: [0.0; 3],
        angmom: [0.0; 3],
        principal_moments,
        principal_axes,
        total_mass,
        inv_mass: if total_mass > 0.0 { 1.0 / total_mass } else { 0.0 },
        force: [0.0; 3],
        torque: [0.0; 3],
        image: [0; 3],
        body_offsets,
        sub_sphere_radii,
        sub_sphere_tags,
    };
    body_store.bodies.push(body);
    body_store.generate_map();

    // Insert sub-sphere atoms (no parent atom)
    for (si, sphere) in def.spheres.iter().enumerate() {
        let sub_tag = base_tag + si as u32;
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
        atoms.vel.push(com_vel);
        atoms.force.push([0.0; 3]);
        atoms.mass.push(sub_mass);
        atoms.inv_mass.push(0.0); // Sub-spheres don't integrate via Verlet
        atoms.cutoff_radius.push(sphere.radius);
        atoms.image.push([0, 0, 0]);
        atoms.is_ghost.push(false);

        dem.radius.push(sphere.radius);
        dem.density.push(density);
        dem.inv_inertia.push(0.0);
        dem.quaternion.push([1.0, 0.0, 0.0, 0.0]);
        dem.omega.push([0.0; 3]);
        dem.ang_mom.push([0.0; 3]);
        dem.torque.push([0.0; 3]);
        dem.body_id.push(clump_id as f64);

        clump_data.body_id.push(clump_id as f64);
        clump_data.body_offset.push(sphere.offset);

        let _ = si; // suppress unused warning
    }

    let n = def.spheres.len();
    atoms.nlocal += n as u32;
    atoms.natoms += n as u64;

    n
}

/// Check if two atoms belong to the same rigid body (for contact exclusion).
#[inline]
pub fn same_body(clump_data: &ClumpAtom, i: usize, j: usize) -> bool {
    if i >= clump_data.body_id.len() || j >= clump_data.body_id.len() {
        return false;
    }
    let ci = clump_data.body_id[i];
    let cj = clump_data.body_id[j];
    ci > 0.0 && cj > 0.0 && (ci - cj).abs() < 0.5
}

/// Check if atom i is a rigid body sub-sphere.
#[inline]
pub fn is_body_atom(clump_data: &ClumpAtom, i: usize) -> bool {
    i < clump_data.body_id.len() && clump_data.body_id[i] > 0.0
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::{Atom, AtomDataRegistry};
    use dem_atom::DemAtom;

    /// Non-overlapping dimer (center distance > r1 + r2) for deterministic tests.
    fn make_dimer_def() -> ClumpDef {
        ClumpDef {
            name: "dimer".to_string(),
            spheres: vec![
                ClumpSphereConfig {
                    offset: [-0.0015, 0.0, 0.0],
                    radius: 0.001,
                },
                ClumpSphereConfig {
                    offset: [0.0015, 0.0, 0.0],
                    radius: 0.001,
                },
            ],
        }
    }

    fn setup_clump_test() -> (Atom, DemAtom, ClumpAtom, MultisphereBodyStore) {
        (Atom::new(), DemAtom::new(), ClumpAtom::new(), MultisphereBodyStore::new())
    }

    #[test]
    fn test_quat_rotate_identity() {
        let q = [1.0, 0.0, 0.0, 0.0];
        let v = [1.0, 2.0, 3.0];
        let result = quat_rotate(q, v);
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!((result[1] - 2.0).abs() < 1e-12);
        assert!((result[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_quat_rotate_90_degrees_z() {
        let angle = std::f64::consts::FRAC_PI_2;
        let half = angle * 0.5;
        let q = [half.cos(), 0.0, 0.0, half.sin()];
        let v = [1.0, 0.0, 0.0];
        let result = quat_rotate(q, v);
        assert!((result[0]).abs() < 1e-12);
        assert!((result[1] - 1.0).abs() < 1e-12);
        assert!((result[2]).abs() < 1e-12);
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

        assert!((mass - expected_mass).abs() < 1e-15);
        assert!(
            (inertia - expected_inertia).abs() / expected_inertia < 1e-12,
            "got {}, expected {}",
            inertia,
            expected_inertia
        );
    }

    #[test]
    fn test_insert_clump_creates_correct_atoms() {
        let (mut atoms, mut dem, mut clump, mut bodies) = setup_clump_test();
        let def = make_dimer_def();

        let count = insert_clump(
            &mut atoms, &mut dem, &mut clump, &mut bodies, &def,
            [0.0, 0.0, 0.0], [0.0; 3], 2500.0, 0, 1,
        );

        assert_eq!(count, 2, "Should insert 2 sub-spheres (no parent atom)");
        assert_eq!(atoms.nlocal, 2);
        assert_eq!(atoms.natoms, 2);
        assert_eq!(bodies.bodies.len(), 1);

        // Both atoms are sub-spheres with real radii
        assert!((dem.radius[0] - 0.001).abs() < 1e-10);
        assert!((dem.radius[1] - 0.001).abs() < 1e-10);

        // Sub-sphere positions offset from COM
        assert!((atoms.pos[0][0] - (-0.0015)).abs() < 1e-10);
        assert!((atoms.pos[1][0] - 0.0015).abs() < 1e-10);

        // Sub-spheres have zero inv_mass
        assert_eq!(atoms.inv_mass[0], 0.0);
        assert_eq!(atoms.inv_mass[1], 0.0);

        // Body has correct mass
        let r = 0.001;
        let m_sphere = 2500.0 * (4.0 / 3.0) * PI * r * r * r;
        assert!(
            (bodies.bodies[0].total_mass - 2.0 * m_sphere).abs() / (2.0 * m_sphere) < 1e-12,
            "mass: got {}, expected {}",
            bodies.bodies[0].total_mass,
            2.0 * m_sphere
        );

        // Body has principal moments (diagonalized)
        assert!(bodies.bodies[0].principal_moments[0] > 0.0);
    }

    #[test]
    fn test_same_body_exclusion() {
        let (mut atoms, mut dem, mut clump, mut bodies) = setup_clump_test();
        let def = make_dimer_def();

        insert_clump(
            &mut atoms, &mut dem, &mut clump, &mut bodies, &def,
            [0.0, 0.0, 0.0], [0.0; 3], 2500.0, 0, 1,
        );

        // Atoms 0 and 1 are in same body
        assert!(same_body(&clump, 0, 1));
        // Backward compat
        assert!(same_body(&clump, 0, 1));
    }

    #[test]
    fn test_different_bodies_not_excluded() {
        let (mut atoms, mut dem, mut clump, mut bodies) = setup_clump_test();
        let def = make_dimer_def();

        insert_clump(
            &mut atoms, &mut dem, &mut clump, &mut bodies, &def,
            [0.0, 0.0, 0.0], [0.0; 3], 2500.0, 0, 1,
        );
        insert_clump(
            &mut atoms, &mut dem, &mut clump, &mut bodies, &def,
            [0.01, 0.0, 0.0], [0.0; 3], 2500.0, 0, 2,
        );

        // Sub-spheres from different bodies not excluded
        assert!(!same_body(&clump, 0, 2)); // body 1 sub vs body 2 sub
        assert!(!same_body(&clump, 1, 3));
    }

    #[test]
    fn test_force_aggregation() {
        let (mut atoms, mut dem, mut clump, mut bodies) = setup_clump_test();
        let def = make_dimer_def();

        insert_clump(
            &mut atoms, &mut dem, &mut clump, &mut bodies, &def,
            [0.0, 0.0, 0.0], [0.0; 3], 2500.0, 0, 1,
        );

        // Apply force to sub-sphere 0 (at x = -0.0015)
        atoms.force[0] = [0.0, 0.0, 10.0];

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(clump);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(registry);
        app.add_resource(bodies);
        app.add_update_system(aggregate_clump_forces, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let atoms = app.get_resource_ref::<Atom>().unwrap();
        let bodies = app.get_resource_ref::<MultisphereBodyStore>().unwrap();

        // Force transferred to body
        assert!(
            (bodies.bodies[0].force[2] - 10.0).abs() < 1e-10,
            "Body z-force should be 10.0, got {}",
            bodies.bodies[0].force[2]
        );

        // Sub-sphere force zeroed
        assert!(
            atoms.force[0][2].abs() < 1e-10,
            "Sub-sphere force should be zeroed"
        );

        // Torque: r × F where r = q * offset = [-0.0015, 0, 0] (identity q)
        // [-0.0015, 0, 0] × [0, 0, 10] = [0, 0.015, 0]
        assert!(
            (bodies.bodies[0].torque[1] - 0.015).abs() < 1e-10,
            "Body y-torque should be 0.015, got {}",
            bodies.bodies[0].torque[1]
        );
    }

    #[test]
    fn test_position_update_after_rotation() {
        let (mut atoms, mut dem, mut clump, mut bodies) = setup_clump_test();
        let def = make_dimer_def();

        insert_clump(
            &mut atoms, &mut dem, &mut clump, &mut bodies, &def,
            [0.0, 0.0, 0.0], [0.0; 3], 2500.0, 0, 1,
        );

        // Rotate body 90° about z-axis
        let angle = std::f64::consts::FRAC_PI_2;
        let half = angle * 0.5;
        bodies.bodies[0].quaternion = [half.cos(), 0.0, 0.0, half.sin()];

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(clump);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(registry);
        app.add_resource(bodies);
        app.add_update_system(update_clump_positions, ScheduleSet::PostFinalIntegration);
        app.organize_systems();
        app.run();

        let atoms = app.get_resource_ref::<Atom>().unwrap();

        // After 90° z rotation:
        // Sub 0 offset [-0.0015, 0, 0] -> [0, -0.0015, 0]
        assert!((atoms.pos[0][0]).abs() < 1e-10);
        assert!((atoms.pos[0][1] - (-0.0015)).abs() < 1e-10);

        // Sub 1 offset [0.0015, 0, 0] -> [0, 0.0015, 0]
        assert!((atoms.pos[1][0]).abs() < 1e-10);
        assert!((atoms.pos[1][1] - 0.0015).abs() < 1e-10);
    }

    #[test]
    fn test_dimer_free_fall() {
        let (mut atoms, mut dem, mut clump, mut bodies) = setup_clump_test();
        let def = make_dimer_def();

        let com_pos = [0.0, 0.0, 0.1];
        insert_clump(
            &mut atoms, &mut dem, &mut clump, &mut bodies, &def,
            com_pos, [0.0; 3], 2500.0, 0, 1,
        );
        atoms.dt = 1e-6;

        let gravity_z = -9.81;
        let total_mass = bodies.bodies[0].total_mass;

        let nsteps = 100;
        let dt = atoms.dt;
        let mut expected_vel_z = 0.0;
        let mut expected_pos_z = com_pos[2];

        for _ in 0..nsteps {
            // Apply gravity to body
            bodies.bodies[0].force = [0.0, 0.0, total_mass * gravity_z];

            // Initial half-step
            body::integrate_body_initial(&mut bodies.bodies[0], dt);

            // Expected trajectory
            expected_vel_z += 0.5 * dt * gravity_z;
            expected_pos_z += expected_vel_z * dt;

            // Final half-step (same force)
            bodies.bodies[0].force = [0.0, 0.0, total_mass * gravity_z];
            body::integrate_body_final(&mut bodies.bodies[0], dt);
            expected_vel_z += 0.5 * dt * gravity_z;
        }

        assert!(
            (bodies.bodies[0].com_pos[2] - expected_pos_z).abs() < 1e-14,
            "COM z: got {}, expected {}",
            bodies.bodies[0].com_pos[2],
            expected_pos_z
        );
    }

    #[test]
    fn test_subsphere_velocity_from_rotation() {
        let (mut atoms, mut dem, mut clump, mut bodies) = setup_clump_test();
        let def = make_dimer_def();

        insert_clump(
            &mut atoms, &mut dem, &mut clump, &mut bodies, &def,
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 2500.0, 0, 1,
        );

        bodies.bodies[0].omega = [0.0, 0.0, 100.0];

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(clump);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(registry);
        app.add_resource(bodies);
        app.add_update_system(update_clump_positions, ScheduleSet::PostFinalIntegration);
        app.organize_systems();
        app.run();

        let atoms = app.get_resource_ref::<Atom>().unwrap();

        // Sub 1 at offset [0.0015, 0, 0]:
        // vel = [1,0,0] + [0,0,100]×[0.0015,0,0] = [1, 0.15, 0]
        assert!((atoms.vel[1][0] - 1.0).abs() < 1e-10);
        assert!((atoms.vel[1][1] - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_contact_on_one_sphere_creates_torque() {
        let (mut atoms, mut dem, mut clump, mut bodies) = setup_clump_test();
        let def = make_dimer_def();

        insert_clump(
            &mut atoms, &mut dem, &mut clump, &mut bodies, &def,
            [0.0, 0.0, 0.0], [0.0; 3], 2500.0, 0, 1,
        );

        // Force in y on right sub-sphere (at x = +0.0015)
        atoms.force[1] = [0.0, 5.0, 0.0];

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(clump);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(registry);
        app.add_resource(bodies);
        app.add_update_system(aggregate_clump_forces, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let bodies = app.get_resource_ref::<MultisphereBodyStore>().unwrap();

        // Force on body
        assert!((bodies.bodies[0].force[1] - 5.0).abs() < 1e-10);

        // Torque: [0.0015, 0, 0] × [0, 5, 0] = [0, 0, 0.0075]
        assert!(
            (bodies.bodies[0].torque[2] - 0.0075).abs() < 1e-10,
            "z-torque should be 0.0075, got {}",
            bodies.bodies[0].torque[2]
        );
    }
}

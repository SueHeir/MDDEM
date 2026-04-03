//! Bond force models for DEM simulations.
//!
//! This crate provides [`DemBondPlugin`], which adds elastic bond forces
//! between particle pairs. Bonds resist relative motion along three
//! independent channels:
//!
//! ## Bond types
//!
//! | Channel    | Stiffness         | Damping          | Description                       |
//! |------------|-------------------|------------------|-----------------------------------|
//! | Normal     | `normal_stiffness`     | `normal_damping`     | Resists stretch/compression along bond axis |
//! | Tangential | `tangential_stiffness` | `tangential_damping` | Resists sliding perpendicular to bond axis  |
//! | Bending    | `bending_stiffness`    | `bending_damping`    | Resists relative rotation between particles |
//!
//! ## Force equations
//!
//! **Normal force** (along the bond axis, unit vector **n̂** from *i* to *j*):
//!
//! ```text
//! F_n = (k_n · δ + γ_n · v_n) · n̂
//! ```
//!
//! where `δ = |r_ij| − r₀` is the stretch and `v_n` is the relative velocity
//! projected onto the bond axis.
//!
//! **Tangential force** (perpendicular to bond axis):
//!
//! ```text
//! F_t = −(k_t · Δs + γ_t · v_t)
//! ```
//!
//! where `Δs` is the accumulated tangential displacement history and `v_t` is the
//! tangential relative velocity. Tangential displacement is rotated each step to
//! stay perpendicular to the current bond orientation.
//!
//! **Bending moment** (resists relative angular velocity **ω_rel**):
//!
//! ```text
//! M = −(k_bend · Δθ + γ_bend · ω_rel)
//! ```
//!
//! where `Δθ` is the accumulated relative rotation angle.
//!
//! ## Breakage criteria
//!
//! Bonds can optionally break when thresholds are exceeded:
//!
//! - **Normal stretch**: bond breaks if `|δ / r₀| > break_normal_stretch`
//! - **Shear displacement**: bond breaks if `|Δs| > break_shear`
//!
//! ## Bond creation
//!
//! Bonds can be created in two ways:
//!
//! - **Auto-bonding**: set `auto_bond = true` to bond all particles whose centers
//!   are within `bond_tolerance × (r_i + r_j)` at the start of the simulation.
//! - **File loading**: set `file = "path.lammps"` and `format = "lammps_data"` to
//!   read bonds from a LAMMPS data file's `Bonds` section.
//!
//! ## TOML configuration
//!
//! All parameters live under the `[bonds]` section:
//!
//! ```toml
//! [bonds]
//! auto_bond = true              # auto-bond touching particles at setup
//! bond_tolerance = 1.001        # multiplier on sum-of-radii for auto-bond
//! normal_stiffness = 1e7        # k_n (N/m)
//! normal_damping = 10.0         # γ_n (N·s/m)
//! tangential_stiffness = 5e6    # k_t (N/m)
//! tangential_damping = 5.0      # γ_t (N·s/m)
//! bending_stiffness = 1e4       # k_bend (N·m/rad)
//! bending_damping = 1.0         # γ_bend (N·m·s/rad)
//! break_normal_stretch = 0.1    # fractional strain threshold (optional)
//! break_shear = 0.0005          # tangential displacement threshold (optional)
//! # file = "bonds.lammps"       # optional: load bonds from file
//! # format = "lammps_data"      # file format (only lammps_data supported)
//! ```

use std::any::Any;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use sim_app::prelude::*;
use sim_scheduler::prelude::*;
use serde::Deserialize;

use dem_atom::DemAtom;
use mddem_core::{Atom, AtomData, AtomDataRegistry, BondEntry, BondStore, CommResource, Config, ParticleSimScheduleSet, ScheduleSetupSet, VirialStress, VirialStressPlugin};
use mddem_print::Thermo;

// ── BondConfig ──────────────────────────────────────────────────────────────

/// Deserialized TOML `[bonds]` configuration section.
///
/// Controls bond stiffness, damping, breakage thresholds, and creation mode.
/// See the [module-level docs](crate) for the full list of parameters and an
/// example TOML block.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct BondConfig {
    /// When `true`, automatically bond all particle pairs whose centre-to-centre
    /// distance is within `bond_tolerance × (r_i + r_j)` during setup.
    /// Default: `false`.
    #[serde(default)]
    pub auto_bond: bool,
    /// Multiplier applied to the sum of radii when checking auto-bond eligibility.
    /// A value slightly above 1.0 prevents missing bonds due to floating-point
    /// round-off. Default: `1.001`.
    #[serde(default = "default_bond_tolerance")]
    pub bond_tolerance: f64,
    /// Normal spring stiffness *k_n* (N/m). Controls resistance to stretch and
    /// compression along the bond axis. Default: `0.0`.
    #[serde(default)]
    pub normal_stiffness: f64,
    /// Normal viscous damping coefficient *γ_n* (N·s/m). Dissipates energy from
    /// relative normal velocity. Default: `0.0`.
    #[serde(default)]
    pub normal_damping: f64,
    /// Tangential spring stiffness *k_t* (N/m). Controls resistance to sliding
    /// perpendicular to the bond axis. Default: `0.0`.
    #[serde(default)]
    pub tangential_stiffness: f64,
    /// Tangential viscous damping coefficient *γ_t* (N·s/m). Default: `0.0`.
    #[serde(default)]
    pub tangential_damping: f64,
    /// Bending stiffness *k_bend* (N·m/rad). Controls resistance to relative
    /// rotation between bonded particles. Default: `0.0`.
    #[serde(default)]
    pub bending_stiffness: f64,
    /// Bending damping coefficient *γ_bend* (N·m·s/rad). Default: `0.0`.
    #[serde(default)]
    pub bending_damping: f64,
    /// Critical normal strain for bond breakage, expressed as a fraction of the
    /// equilibrium length *r₀* (e.g. `0.1` = 10% strain). When `Some`, bonds
    /// break if `|δ / r₀| > break_normal_stretch`. Default: `None` (unbreakable).
    #[serde(default)]
    pub break_normal_stretch: Option<f64>,
    /// Critical tangential displacement magnitude for bond breakage (m). When
    /// `Some`, bonds break if `|Δs| > break_shear`. Default: `None` (unbreakable).
    #[serde(default)]
    pub break_shear: Option<f64>,
    /// Path to a LAMMPS data file containing a `Bonds` section to load at setup.
    pub file: Option<String>,
    /// File format identifier. Currently only `"lammps_data"` is supported.
    pub format: Option<String>,
}

fn default_bond_tolerance() -> f64 {
    1.001
}

impl Default for BondConfig {
    fn default() -> Self {
        BondConfig {
            auto_bond: false,
            bond_tolerance: 1.001,
            normal_stiffness: 0.0,
            normal_damping: 0.0,
            tangential_stiffness: 0.0,
            tangential_damping: 0.0,
            bending_stiffness: 0.0,
            bending_damping: 0.0,
            break_normal_stretch: None,
            break_shear: None,
            file: None,
            format: None,
        }
    }
}

// ── BondHistoryStore ────────────────────────────────────────────────────────

/// Per-bond history tracking tangential displacement and bending angle.
///
/// Each bonded pair maintains its own `BondHistoryEntry` so that incremental
/// tangential and bending springs can be integrated across time steps.
#[derive(Clone, Debug)]
pub struct BondHistoryEntry {
    /// Global tag of the bonded partner particle.
    pub partner_tag: u32,
    /// Accumulated tangential spring displacement vector **Δs** (m).
    /// Rotated each step to stay perpendicular to the current bond axis.
    pub delta_t: [f64; 3],
    /// Accumulated relative rotation angle vector **Δθ** (rad).
    pub delta_theta: [f64; 3],
}

/// Per-atom storage of [`BondHistoryEntry`] lists, kept in sync with
/// [`BondStore`](mddem_core::BondStore).
///
/// Implements [`AtomData`] for MPI communication and atom reordering.
pub struct BondHistoryStore {
    /// Outer index = local atom index, inner = one entry per bond on that atom.
    pub history: Vec<Vec<BondHistoryEntry>>,
}

impl BondHistoryStore {
    /// Creates an empty bond history store with no atom entries.
    pub fn new() -> Self {
        BondHistoryStore {
            history: Vec::new(),
        }
    }
}

impl AtomData for BondHistoryStore {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn truncate(&mut self, n: usize) {
        self.history.resize_with(n, Vec::new);
        self.history.truncate(n);
    }

    fn swap_remove(&mut self, i: usize) {
        if i < self.history.len() {
            self.history.swap_remove(i);
        }
    }

    fn apply_permutation(&mut self, perm: &[usize], n: usize) {
        let new_history: Vec<Vec<BondHistoryEntry>> =
            perm.iter().map(|&p| self.history[p].clone()).collect();
        self.history[..n].clone_from_slice(&new_history);
    }

    /// Pack bond history for atom `i` into `buf`.
    ///
    /// Wire format: `[count, (partner_tag, dt[3], dθ[3]) × count]` — `1 + 7 × count` f64s.
    fn pack(&self, i: usize, buf: &mut Vec<f64>) {
        if i < self.history.len() {
            let list = &self.history[i];
            buf.push(list.len() as f64);
            for entry in list {
                buf.push(entry.partner_tag as f64);
                buf.push(entry.delta_t[0]);
                buf.push(entry.delta_t[1]);
                buf.push(entry.delta_t[2]);
                buf.push(entry.delta_theta[0]);
                buf.push(entry.delta_theta[1]);
                buf.push(entry.delta_theta[2]);
            }
        } else {
            buf.push(0.0);
        }
    }

    fn unpack(&mut self, buf: &[f64]) -> usize {
        let count = buf[0] as usize;
        let mut list = Vec::with_capacity(count);
        let mut pos = 1;
        for _ in 0..count {
            let partner_tag = buf[pos] as u32;
            let delta_t = [buf[pos + 1], buf[pos + 2], buf[pos + 3]];
            let delta_theta = [buf[pos + 4], buf[pos + 5], buf[pos + 6]];
            list.push(BondHistoryEntry {
                partner_tag,
                delta_t,
                delta_theta,
            });
            pos += 7;
        }
        self.history.push(list);
        pos
    }
}

// ── BondMetrics ─────────────────────────────────────────────────────────────

/// Accumulated per-step bond metrics, exposed via the thermo system.
///
/// Reset each step in [`zero_bond_metrics`] and populated during
/// [`bond_force`]. The average strain and cumulative breakage count are
/// written to [`Thermo`] in [`output_bond_metrics`].
#[derive(Default)]
pub struct BondMetrics {
    /// Sum of `δ / r₀` over all active bonds this step.
    pub strain_sum: f64,
    /// Number of active bonds evaluated this step.
    pub bond_count: usize,
    /// Number of bonds broken during this step.
    pub bonds_broken_this_step: usize,
    /// Cumulative number of bonds broken since the start of the simulation.
    pub total_bonds_broken: usize,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Plugin that enables elastic bond forces between DEM particles.
///
/// Registers the [`BondStore`](mddem_core::BondStore), [`BondHistoryStore`],
/// and [`BondMetrics`] resources, and adds setup systems for auto-bonding and
/// file-based bond loading, plus per-step force computation with optional
/// breakage.
pub struct DemBondPlugin;

impl Plugin for DemBondPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[bonds]
# auto_bond = false
# bond_tolerance = 1.001
# normal_stiffness = 0.0
# normal_damping = 0.0
# tangential_stiffness = 0.0
# tangential_damping = 0.0
# bending_stiffness = 0.0
# bending_damping = 0.0
# break_normal_stretch = 0.1   # optional: fractional strain to break
# break_shear = 0.0005         # optional: tangential displacement to break"#,
        )
    }

    fn build(&self, app: &mut App) {
        app.add_plugins(mddem_core::BondPlugin);
        app.add_plugins(VirialStressPlugin);
        Config::load::<BondConfig>(app, "bonds");
        app.add_resource(BondMetrics::default());
        mddem_core::register_atom_data!(app, BondHistoryStore::new());
        app.add_setup_system(
            auto_bond_touching.run_if(first_stage_only()),
            ScheduleSetupSet::PostSetup,
        );
        app.add_setup_system(
            load_bonds_from_file.run_if(first_stage_only()),
            ScheduleSetupSet::PostSetup,
        );
        app.add_setup_system(
            init_bond_history.run_if(first_stage_only()),
            ScheduleSetupSet::PostSetup,
        );
        app.add_update_system(zero_bond_metrics, ParticleSimScheduleSet::PreForce);
        app.add_update_system(bond_force.label("dem_bond_force"), ParticleSimScheduleSet::Force);
        app.add_update_system(output_bond_metrics, ParticleSimScheduleSet::PostForce);
    }
}

// ── Setup systems ───────────────────────────────────────────────────────────

/// Auto-bond initially touching particles at setup time.
///
/// For every pair of local atoms whose centre-to-centre distance is within
/// `bond_tolerance × (r_i + r_j)`, a symmetric bond entry is created in
/// both atoms' bond lists with the current separation as the equilibrium
/// length *r₀*.
pub fn auto_bond_touching(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    bond_config: Res<BondConfig>,
    comm: Res<CommResource>,
) {
    if !bond_config.auto_bond {
        return;
    }

    let dem = registry.expect::<DemAtom>("auto_bond_touching");
    let mut bond_store = registry.expect_mut::<BondStore>("auto_bond_touching");

    let nlocal = atoms.nlocal as usize;

    while bond_store.bonds.len() < nlocal {
        bond_store.bonds.push(Vec::new());
    }

    let tolerance = bond_config.bond_tolerance;
    let mut bond_count = 0u64;

    for i in 0..nlocal {
        for j in (i + 1)..nlocal {
            let dx = atoms.pos[j][0] - atoms.pos[i][0];
            let dy = atoms.pos[j][1] - atoms.pos[i][1];
            let dz = atoms.pos[j][2] - atoms.pos[i][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            let sum_radii = dem.radius[i] + dem.radius[j];
            if dist <= sum_radii * tolerance {
                bond_store.bonds[i].push(BondEntry {
                    partner_tag: atoms.tag[j],
                    bond_type: 0,
                    r0: dist,
                });
                bond_store.bonds[j].push(BondEntry {
                    partner_tag: atoms.tag[i],
                    bond_type: 0,
                    r0: dist,
                });
                bond_count += 1;
            }
        }
    }

    if comm.rank() == 0 {
        println!("DemBond: auto-bonded {} pairs", bond_count);
    }
}

/// Load bonds from a LAMMPS data file at setup time.
///
/// Parses the `Bonds` section of a LAMMPS data file. Each line in that
/// section has the format `bond_id bond_type atom1_tag atom2_tag`.
/// The equilibrium length *r₀* is computed from the atoms' current positions.
pub fn load_bonds_from_file(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    bond_config: Res<BondConfig>,
    comm: Res<CommResource>,
) {
    let file_path = match bond_config.file.as_deref() {
        Some(p) => p,
        None => return,
    };

    let format = bond_config.format.as_deref().unwrap_or("lammps_data");
    if format != "lammps_data" {
        eprintln!(
            "ERROR: Unsupported bond file format '{}'. Supported: lammps_data",
            format
        );
        std::process::exit(1);
    }

    let file = File::open(file_path).unwrap_or_else(|e| {
        eprintln!("ERROR: Failed to open bond file '{}': {}", file_path, e);
        std::process::exit(1);
    });
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader
        .lines()
        .map(|l| l.expect("failed to read line from bond file"))
        .collect();

    let section_headers = [
        "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
        "Masses", "Pair Coeffs",
    ];
    let is_section_header = |line: &str| -> bool {
        let trimmed = line.trim();
        section_headers.iter().any(|h| trimmed.starts_with(h))
    };

    let mut bonds_start = None;
    for (i, line) in lines.iter().enumerate() {
        if line.trim().starts_with("Bonds") {
            bonds_start = Some(i + 1);
        }
    }

    let bonds_start = match bonds_start {
        Some(s) => s,
        None => {
            if comm.rank() == 0 {
                println!("DemBond: no Bonds section found in '{}', skipping", file_path);
            }
            return;
        }
    };

    let nlocal = atoms.nlocal as usize;
    let mut tag_to_local: HashMap<u32, usize> = HashMap::with_capacity(nlocal);
    for i in 0..nlocal {
        tag_to_local.insert(atoms.tag[i], i);
    }

    let mut bond_store = registry.expect_mut::<BondStore>("load_bonds_from_file");

    while bond_store.bonds.len() < nlocal {
        bond_store.bonds.push(Vec::new());
    }

    let mut bond_count = 0u64;

    for i in bonds_start..lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() {
            continue;
        }
        if is_section_header(trimmed) {
            break;
        }
        if trimmed.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() < 4 {
            continue;
        }

        let bond_type: u32 = fields[1].parse().unwrap_or(0);
        let tag1: u32 = fields[2]
            .parse()
            .expect("failed to parse atom tag1 in Bonds section");
        let tag2: u32 = fields[3]
            .parse()
            .expect("failed to parse atom tag2 in Bonds section");

        let idx1 = match tag_to_local.get(&tag1) {
            Some(&i) => i,
            None => continue,
        };
        let idx2 = match tag_to_local.get(&tag2) {
            Some(&i) => i,
            None => continue,
        };

        let dx = atoms.pos[idx2][0] - atoms.pos[idx1][0];
        let dy = atoms.pos[idx2][1] - atoms.pos[idx1][1];
        let dz = atoms.pos[idx2][2] - atoms.pos[idx1][2];
        let r0 = (dx * dx + dy * dy + dz * dz).sqrt();

        bond_store.bonds[idx1].push(BondEntry {
            partner_tag: tag2,
            bond_type,
            r0,
        });
        bond_store.bonds[idx2].push(BondEntry {
            partner_tag: tag1,
            bond_type,
            r0,
        });
        bond_count += 1;
    }

    if comm.rank() == 0 {
        println!(
            "DemBond: loaded {} bonds from LAMMPS data file '{}'",
            bond_count, file_path
        );
    }
}

/// Initialize bond history entries to match the current bond store.
///
/// For every bond that does not yet have a corresponding history entry, a
/// zero-initialized [`BondHistoryEntry`] is created. This is called once
/// during setup after bonds have been created or loaded.
pub fn init_bond_history(registry: Res<AtomDataRegistry>) {
    let bond_store = registry.get::<BondStore>();
    let bonds = match bond_store {
        Some(ref b) => b,
        None => return,
    };

    let mut history = registry.expect_mut::<BondHistoryStore>("init_bond_history");

    // Ensure history covers all atoms with matching entries
    while history.history.len() < bonds.bonds.len() {
        history.history.push(Vec::new());
    }

    for i in 0..bonds.bonds.len() {
        // Only add entries for bonds that don't have history yet
        for bond in &bonds.bonds[i] {
            let has_entry = history.history[i]
                .iter()
                .any(|h| h.partner_tag == bond.partner_tag);
            if !has_entry {
                history.history[i].push(BondHistoryEntry {
                    partner_tag: bond.partner_tag,
                    delta_t: [0.0; 3],
                    delta_theta: [0.0; 3],
                });
            }
        }
    }
}

// ── Force systems ───────────────────────────────────────────────────────────

/// Computes normal, tangential, and bending bond forces for all local atoms.
///
/// Iterates over every bond owned by each local atom, computing elastic spring
/// and viscous damping contributions. Each bond is processed once (lower tag
/// owns the computation). Bonds exceeding breakage thresholds are collected and
/// removed after the force loop completes.
pub fn bond_force(
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    bond_config: Res<BondConfig>,
    mut metrics: ResMut<BondMetrics>,
    mut virial: Option<ResMut<VirialStress>>,
) {
    let bond_store = registry.get::<BondStore>();
    let bonds = match bond_store {
        Some(ref b) => b,
        None => return,
    };

    let k_n = bond_config.normal_stiffness;
    let gamma_n = bond_config.normal_damping;
    let k_t = bond_config.tangential_stiffness;
    let gamma_t = bond_config.tangential_damping;
    let k_bend = bond_config.bending_stiffness;
    let gamma_bend = bond_config.bending_damping;

    if k_n == 0.0 && gamma_n == 0.0 && k_t == 0.0 && k_bend == 0.0 {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    if bonds.bonds.len() < nlocal {
        return;
    }


    let has_tangential = k_t > 0.0 || gamma_t > 0.0;
    let has_bending = k_bend > 0.0 || gamma_bend > 0.0;
    let has_history = has_tangential || has_bending;

    // Get DemAtom and BondHistoryStore if needed for tangential/bending
    let mut dem_opt = if has_history {
        Some(registry.expect_mut::<DemAtom>("bond_force"))
    } else {
        None
    };
    let mut hist_opt = if has_history {
        Some(registry.expect_mut::<BondHistoryStore>("bond_force"))
    } else {
        None
    };

    let dt = atoms.dt;

    // Build tag → index lookup for all atoms (local + ghost)
    let mut tag_to_index: HashMap<u32, usize> = HashMap::with_capacity(atoms.len());
    for idx in 0..atoms.len() {
        tag_to_index.insert(atoms.tag[idx], idx);
    }

    // Collect bonds to break (deferred removal)
    let mut bonds_to_break: Vec<(u32, u32)> = Vec::new();

    for i in 0..nlocal {
        for b_idx in 0..bonds.bonds[i].len() {
            let bond = &bonds.bonds[i][b_idx];
            let j = match tag_to_index.get(&bond.partner_tag) {
                Some(&idx) => idx,
                None => continue,
            };

            // Process each bond once: only when i's tag < partner's tag
            if atoms.tag[i] >= bond.partner_tag {
                continue;
            }

            let dx = atoms.pos[j][0] - atoms.pos[i][0];
            let dy = atoms.pos[j][1] - atoms.pos[i][1];
            let dz = atoms.pos[j][2] - atoms.pos[i][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist < 1e-20 {
                continue;
            }

            // Unit normal from i to j
            let nx = dx / dist;
            let ny = dy / dist;
            let nz = dz / dist;

            // Normal stretch
            let delta = dist - bond.r0;

            // Check normal breaking
            if let Some(break_stretch) = bond_config.break_normal_stretch {
                let strain = (delta / bond.r0).abs();
                if strain > break_stretch {
                    bonds_to_break.push((atoms.tag[i], bond.partner_tag));
                    continue;
                }
            }

            // Normal spring force: F_spring = k_n · δ  (positive = tension)
            let f_spring = k_n * delta;

            // Relative velocity of j w.r.t. i, projected onto bond axis
            let dvx = atoms.vel[j][0] - atoms.vel[i][0];
            let dvy = atoms.vel[j][1] - atoms.vel[i][1];
            let dvz = atoms.vel[j][2] - atoms.vel[i][2];
            let v_n = dvx * nx + dvy * ny + dvz * nz;

            // Total normal force: spring + viscous damping, applied along n̂
            let f_damp = gamma_n * v_n;
            let f_total_n = f_spring + f_damp;

            let fx = f_total_n * nx;
            let fy = f_total_n * ny;
            let fz = f_total_n * nz;

            atoms.force[i][0] += fx;
            atoms.force[i][1] += fy;
            atoms.force[i][2] += fz;
            atoms.force[j][0] -= fx;
            atoms.force[j][1] -= fy;
            atoms.force[j][2] -= fz;

            // Virial
            if let Some(ref mut v) = virial {
                if v.active {
                    v.add_pair(dx, dy, dz, fx, fy, fz);
                }
            }

            // Tangential force
            if has_tangential {
                let history = hist_opt.as_mut().unwrap();

                // Find history entry for this bond
                let h_idx = history.history[i]
                    .iter()
                    .position(|h| h.partner_tag == bond.partner_tag);

                // Tangential relative velocity: v_t = v_rel − (v_rel · n̂) n̂
                let vt_x = dvx - v_n * nx;
                let vt_y = dvy - v_n * ny;
                let vt_z = dvz - v_n * nz;

                // Get or create history
                let (dt_x, dt_y, dt_z) = if let Some(idx) = h_idx {
                    let h = &mut history.history[i][idx];
                    // Rotate spring to current bond direction
                    let s_dot_n = h.delta_t[0] * nx + h.delta_t[1] * ny + h.delta_t[2] * nz;
                    h.delta_t[0] -= s_dot_n * nx;
                    h.delta_t[1] -= s_dot_n * ny;
                    h.delta_t[2] -= s_dot_n * nz;
                    // Integrate
                    h.delta_t[0] += vt_x * dt;
                    h.delta_t[1] += vt_y * dt;
                    h.delta_t[2] += vt_z * dt;
                    (h.delta_t[0], h.delta_t[1], h.delta_t[2])
                } else {
                    let new_dt = [vt_x * dt, vt_y * dt, vt_z * dt];
                    history.history[i].push(BondHistoryEntry {
                        partner_tag: bond.partner_tag,
                        delta_t: new_dt,
                        delta_theta: [0.0; 3],
                    });
                    (new_dt[0], new_dt[1], new_dt[2])
                };

                // Check shear breaking
                let shear_mag = (dt_x * dt_x + dt_y * dt_y + dt_z * dt_z).sqrt();

                if let Some(break_shear) = bond_config.break_shear {
                    if shear_mag > break_shear {
                        bonds_to_break.push((atoms.tag[i], bond.partner_tag));
                        continue;
                    }
                }

                // Tangential force: F_t = −(k_t · Δs + γ_t · v_t)
                let ft_x = -(k_t * dt_x + gamma_t * vt_x);
                let ft_y = -(k_t * dt_y + gamma_t * vt_y);
                let ft_z = -(k_t * dt_z + gamma_t * vt_z);

                atoms.force[i][0] += ft_x;
                atoms.force[i][1] += ft_y;
                atoms.force[i][2] += ft_z;
                atoms.force[j][0] -= ft_x;
                atoms.force[j][1] -= ft_y;
                atoms.force[j][2] -= ft_z;

                // Torques from tangential bond force: τ = r × F_t
                // Lever arm is half the equilibrium length along the bond axis.
                // Both particles receive the same torque direction because the
                // lever arms and forces are both flipped: τ_j = (−r) × (−F_t) = r × F_t
                let half_r0 = bond.r0 * 0.5;
                let rn_x = half_r0 * nx;
                let rn_y = half_r0 * ny;
                let rn_z = half_r0 * nz;
                let ti_x = rn_y * ft_z - rn_z * ft_y;
                let ti_y = rn_z * ft_x - rn_x * ft_z;
                let ti_z = rn_x * ft_y - rn_y * ft_x;

                if let Some(ref mut dem) = dem_opt {
                    dem.torque[i][0] += ti_x;
                    dem.torque[i][1] += ti_y;
                    dem.torque[i][2] += ti_z;
                    dem.torque[j][0] += ti_x; // same direction (both arms × same force)
                    dem.torque[j][1] += ti_y;
                    dem.torque[j][2] += ti_z;
                }
            }

            // Bending moment
            if has_bending {
                let dem = dem_opt.as_ref().unwrap();
                let history = hist_opt.as_mut().unwrap();

                let omega_rel_x = dem.omega[j][0] - dem.omega[i][0];
                let omega_rel_y = dem.omega[j][1] - dem.omega[i][1];
                let omega_rel_z = dem.omega[j][2] - dem.omega[i][2];

                // Find history entry
                let h_idx = history.history[i]
                    .iter()
                    .position(|h| h.partner_tag == bond.partner_tag);

                let (dth_x, dth_y, dth_z) = if let Some(idx) = h_idx {
                    let h = &mut history.history[i][idx];
                    h.delta_theta[0] += omega_rel_x * dt;
                    h.delta_theta[1] += omega_rel_y * dt;
                    h.delta_theta[2] += omega_rel_z * dt;
                    (h.delta_theta[0], h.delta_theta[1], h.delta_theta[2])
                } else {
                    let new_dth = [omega_rel_x * dt, omega_rel_y * dt, omega_rel_z * dt];
                    history.history[i].push(BondHistoryEntry {
                        partner_tag: bond.partner_tag,
                        delta_t: [0.0; 3],
                        delta_theta: new_dth,
                    });
                    (new_dth[0], new_dth[1], new_dth[2])
                };

                // Bending moment: M = −(k_bend · Δθ + γ_bend · ω_rel)
                // Applied as +M to atom i and −M to atom j (equal and opposite).
                let mx = -(k_bend * dth_x + gamma_bend * omega_rel_x);
                let my = -(k_bend * dth_y + gamma_bend * omega_rel_y);
                let mz = -(k_bend * dth_z + gamma_bend * omega_rel_z);

                if let Some(ref mut dem) = dem_opt {
                    dem.torque[i][0] += mx;
                    dem.torque[i][1] += my;
                    dem.torque[i][2] += mz;
                    dem.torque[j][0] -= mx;
                    dem.torque[j][1] -= my;
                    dem.torque[j][2] -= mz;
                }
            }

            // Accumulate bond metrics
            metrics.strain_sum += delta / bond.r0;
            metrics.bond_count += 1;
        }
    }

    // Deferred bond removal — drop all registry borrows first
    if !bonds_to_break.is_empty() {
        drop(bond_store);
        drop(dem_opt);
        drop(hist_opt);

        let mut bond_store = registry.expect_mut::<BondStore>("bond_force_break");
        let mut history_store = if has_history {
            Some(registry.expect_mut::<BondHistoryStore>("bond_force_break"))
        } else {
            None
        };

        for (tag_a, tag_b) in &bonds_to_break {
            for idx in 0..atoms.len() {
                if atoms.tag[idx] == *tag_a || atoms.tag[idx] == *tag_b {
                    let partner = if atoms.tag[idx] == *tag_a { *tag_b } else { *tag_a };
                    if idx < bond_store.bonds.len() {
                        bond_store.bonds[idx].retain(|b| b.partner_tag != partner);
                    }
                    if let Some(ref mut hs) = history_store {
                        if idx < hs.history.len() {
                            hs.history[idx].retain(|h| h.partner_tag != partner);
                        }
                    }
                }
            }
        }

        metrics.bonds_broken_this_step += bonds_to_break.len();
        metrics.total_bonds_broken += bonds_to_break.len();
    }
}

/// Reset per-step bond metrics to zero before the force computation pass.
pub fn zero_bond_metrics(mut metrics: ResMut<BondMetrics>) {
    metrics.strain_sum = 0.0;
    metrics.bond_count = 0;
    metrics.bonds_broken_this_step = 0;
}

/// Write bond metrics to thermo output after force computation.
///
/// Publishes `bond_strain` (average `δ/r₀` across all bonds) and
/// `bonds_broken` (cumulative count) to the [`Thermo`] resource.
pub fn output_bond_metrics(
    metrics: Res<BondMetrics>,
    comm: Res<CommResource>,
    mut thermo: Option<ResMut<Thermo>>,
) {
    let strain_sum = comm.all_reduce_sum_f64(metrics.strain_sum);
    let bond_count = comm.all_reduce_sum_f64(metrics.bond_count as f64);

    if let Some(ref mut thermo) = thermo {
        if bond_count > 0.0 {
            let avg_strain = strain_sum / bond_count;
            thermo.set("bond_strain", avg_strain);
        } else {
            thermo.set("bond_strain", 0.0);
        }
        thermo.set("bonds_broken", metrics.total_bonds_broken as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dem_atom::DemAtom;
    use mddem_core::{Atom, AtomDataRegistry, BondEntry, BondStore, CommResource, SingleProcessComm, toml};
    use mddem_test_utils::push_dem_test_atom;

    fn make_bond_config() -> BondConfig {
        BondConfig {
            auto_bond: false,
            bond_tolerance: 1.001,
            normal_stiffness: 1e7,
            normal_damping: 0.0,
            tangential_stiffness: 0.0,
            tangential_damping: 0.0,
            bending_stiffness: 0.0,
            bending_damping: 0.0,
            break_normal_stretch: None,
            break_shear: None,
            file: None,
            format: None,
        }
    }

    #[test]
    fn auto_bond_creates_symmetric_bonds() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        push_dem_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_dem_test_atom(&mut atom, &mut dem, 2, [0.002, 0.0, 0.0], radius);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(BondStore::new());
        registry.register(BondHistoryStore::new());

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(BondConfig {
            auto_bond: true,
            ..make_bond_config()
        });
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(SchedulerManager::default());
        app.add_setup_system(auto_bond_touching, ScheduleSetupSet::PostSetup);
        app.organize_systems();
        app.setup();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let bonds = registry.expect::<BondStore>("test");
        assert_eq!(bonds.bonds.len(), 2);
        assert_eq!(bonds.bonds[0].len(), 1);
        assert_eq!(bonds.bonds[1].len(), 1);
        assert_eq!(bonds.bonds[0][0].partner_tag, 2);
        assert_eq!(bonds.bonds[1][0].partner_tag, 1);
    }

    #[test]
    fn auto_bond_skips_separated_atoms() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        push_dem_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_dem_test_atom(&mut atom, &mut dem, 2, [0.01, 0.0, 0.0], radius);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(BondStore::new());
        registry.register(BondHistoryStore::new());

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(BondConfig {
            auto_bond: true,
            ..make_bond_config()
        });
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(SchedulerManager::default());
        app.add_setup_system(auto_bond_touching, ScheduleSetupSet::PostSetup);
        app.organize_systems();
        app.setup();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let bonds = registry.expect::<BondStore>("test");
        assert_eq!(bonds.bonds[0].len(), 0);
        assert_eq!(bonds.bonds[1].len(), 0);
    }

    #[test]
    fn bond_force_attracts_stretched_pair() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        push_dem_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_dem_test_atom(&mut atom, &mut dem, 2, [0.0025, 0.0, 0.0], radius);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry { partner_tag: 2, bond_type: 0, r0: 0.002 }]);
        bond_store.bonds.push(vec![BondEntry { partner_tag: 1, bond_type: 0, r0: 0.002 }]);

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(bond_store);
        registry.register(BondHistoryStore::new());

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_bond_config());
        app.add_resource(BondMetrics::default());
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(Thermo::new());
        app.add_update_system(bond_force, ParticleSimScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(atom.force[0][0] > 0.0, "stretched bond attracts atom 0");
        assert!(atom.force[1][0] < 0.0, "stretched bond attracts atom 1");
        assert!((atom.force[0][0] + atom.force[1][0]).abs() < 1e-10);
    }

    #[test]
    fn bond_force_repels_compressed_pair() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        push_dem_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_dem_test_atom(&mut atom, &mut dem, 2, [0.0015, 0.0, 0.0], radius);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry { partner_tag: 2, bond_type: 0, r0: 0.002 }]);
        bond_store.bonds.push(vec![BondEntry { partner_tag: 1, bond_type: 0, r0: 0.002 }]);

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(bond_store);
        registry.register(BondHistoryStore::new());

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_bond_config());
        app.add_resource(BondMetrics::default());
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(Thermo::new());
        app.add_update_system(bond_force, ParticleSimScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(atom.force[0][0] < 0.0, "compressed bond repels atom 0");
        assert!(atom.force[1][0] > 0.0, "compressed bond repels atom 1");
    }

    #[test]
    fn bond_force_zero_at_equilibrium() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        push_dem_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_dem_test_atom(&mut atom, &mut dem, 2, [0.002, 0.0, 0.0], radius);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry { partner_tag: 2, bond_type: 0, r0: 0.002 }]);
        bond_store.bonds.push(vec![BondEntry { partner_tag: 1, bond_type: 0, r0: 0.002 }]);

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(bond_store);
        registry.register(BondHistoryStore::new());

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_bond_config());
        app.add_resource(BondMetrics::default());
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(Thermo::new());
        app.add_update_system(bond_force, ParticleSimScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(atom.force[0][0].abs() < 1e-10);
        assert!(atom.force[1][0].abs() < 1e-10);
    }

    #[test]
    fn tangential_bond_force_perpendicular() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;
        atom.dt = 1e-6;

        push_dem_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_dem_test_atom(&mut atom, &mut dem, 2, [0.002, 0.0, 0.0], radius);
        // Give atom 1 a tangential velocity
        atom.vel[1][1] = 0.1;
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry { partner_tag: 2, bond_type: 0, r0: 0.002 }]);
        bond_store.bonds.push(vec![BondEntry { partner_tag: 1, bond_type: 0, r0: 0.002 }]);

        let mut history = BondHistoryStore::new();
        history.history.push(vec![BondHistoryEntry { partner_tag: 2, delta_t: [0.0; 3], delta_theta: [0.0; 3] }]);
        history.history.push(vec![BondHistoryEntry { partner_tag: 1, delta_t: [0.0; 3], delta_theta: [0.0; 3] }]);

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(bond_store);
        registry.register(history);

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(BondConfig {
            tangential_stiffness: 5e6,
            tangential_damping: 5.0,
            ..make_bond_config()
        });
        app.add_resource(BondMetrics::default());
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(Thermo::new());
        app.add_update_system(bond_force, ParticleSimScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Tangential force should be in y-direction
        assert!(atom.force[0][1].abs() > 0.0, "tangential force on atom 0");
        assert!(
            (atom.force[0][1] + atom.force[1][1]).abs() < 1e-10,
            "Newton's 3rd law for tangential"
        );
    }

    #[test]
    fn bending_torque_opposes_relative_rotation() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;
        atom.dt = 1e-6;

        push_dem_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_dem_test_atom(&mut atom, &mut dem, 2, [0.002, 0.0, 0.0], radius);
        // Give atom 1 angular velocity
        dem.omega[1] = [0.0, 100.0, 0.0];
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry { partner_tag: 2, bond_type: 0, r0: 0.002 }]);
        bond_store.bonds.push(vec![BondEntry { partner_tag: 1, bond_type: 0, r0: 0.002 }]);

        let mut history = BondHistoryStore::new();
        history.history.push(vec![BondHistoryEntry { partner_tag: 2, delta_t: [0.0; 3], delta_theta: [0.0; 3] }]);
        history.history.push(vec![BondHistoryEntry { partner_tag: 1, delta_t: [0.0; 3], delta_theta: [0.0; 3] }]);

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(bond_store);
        registry.register(history);

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(BondConfig {
            bending_stiffness: 1e4,
            bending_damping: 1.0,
            ..make_bond_config()
        });
        app.add_resource(BondMetrics::default());
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(Thermo::new());
        app.add_update_system(bond_force, ParticleSimScheduleSet::Force);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        // Bending torque on atom 0 should be in +y (opposing relative omega of atom 1 - atom 0)
        // omega_rel_y = 100 - 0 = 100, M_y = -(k_bend * dtheta_y + gamma_bend * omega_rel_y)
        // dtheta_y = 100 * dt = 1e-4, M_y = -(1e4 * 1e-4 + 1.0 * 100) = -(1 + 100) = -101
        // Applied to atom 0: +M, applied to atom 1: -M
        assert!(
            dem.torque[0][1] < 0.0,
            "bending moment on atom 0 should oppose relative omega, got {}",
            dem.torque[0][1]
        );
        assert!(
            dem.torque[1][1] > 0.0,
            "bending moment on atom 1 should be opposite, got {}",
            dem.torque[1][1]
        );
    }

    #[test]
    fn bond_breaks_on_normal_stretch() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Stretched well beyond 10%
        push_dem_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_dem_test_atom(&mut atom, &mut dem, 2, [0.003, 0.0, 0.0], radius);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry { partner_tag: 2, bond_type: 0, r0: 0.002 }]);
        bond_store.bonds.push(vec![BondEntry { partner_tag: 1, bond_type: 0, r0: 0.002 }]);

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(bond_store);
        registry.register(BondHistoryStore::new());

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(BondConfig {
            break_normal_stretch: Some(0.1),
            ..make_bond_config()
        });
        app.add_resource(BondMetrics::default());
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(Thermo::new());
        app.add_update_system(bond_force, ParticleSimScheduleSet::Force);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let bonds = registry.expect::<BondStore>("test");
        // Bond should be removed from both atoms
        assert_eq!(bonds.bonds[0].len(), 0, "bond should be broken on atom 0");
        assert_eq!(bonds.bonds[1].len(), 0, "bond should be broken on atom 1");

        let metrics = app.get_resource_ref::<BondMetrics>().unwrap();
        assert_eq!(metrics.bonds_broken_this_step, 1);
        assert_eq!(metrics.total_bonds_broken, 1);
    }

    #[test]
    fn bond_no_break_within_threshold() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Slightly stretched (5% < 10% threshold)
        push_dem_test_atom(&mut atom, &mut dem, 1, [0.0, 0.0, 0.0], radius);
        push_dem_test_atom(&mut atom, &mut dem, 2, [0.0021, 0.0, 0.0], radius);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry { partner_tag: 2, bond_type: 0, r0: 0.002 }]);
        bond_store.bonds.push(vec![BondEntry { partner_tag: 1, bond_type: 0, r0: 0.002 }]);

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);
        registry.register(bond_store);
        registry.register(BondHistoryStore::new());

        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(BondConfig {
            break_normal_stretch: Some(0.1),
            ..make_bond_config()
        });
        app.add_resource(BondMetrics::default());
        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        app.add_resource(Thermo::new());
        app.add_update_system(bond_force, ParticleSimScheduleSet::Force);
        app.organize_systems();
        app.run();

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let bonds = registry.expect::<BondStore>("test");
        assert_eq!(bonds.bonds[0].len(), 1, "bond should not break within threshold");
    }

    #[test]
    fn bond_history_pack_unpack_round_trip() {
        let mut store = BondHistoryStore::new();
        store.history.push(vec![
            BondHistoryEntry {
                partner_tag: 5,
                delta_t: [0.1, 0.2, 0.3],
                delta_theta: [0.4, 0.5, 0.6],
            },
            BondHistoryEntry {
                partner_tag: 10,
                delta_t: [1.0, 2.0, 3.0],
                delta_theta: [4.0, 5.0, 6.0],
            },
        ]);

        let mut buf = Vec::new();
        store.pack(0, &mut buf);
        assert_eq!(buf.len(), 1 + 7 * 2);

        let mut store2 = BondHistoryStore::new();
        let consumed = store2.unpack(&buf);
        assert_eq!(consumed, buf.len());
        assert_eq!(store2.history[0].len(), 2);
        assert_eq!(store2.history[0][0].partner_tag, 5);
        assert!((store2.history[0][0].delta_t[0] - 0.1).abs() < 1e-15);
        assert!((store2.history[0][1].delta_theta[2] - 6.0).abs() < 1e-15);
    }

    #[test]
    fn bond_config_deserialization() {
        let toml_str = r#"
normal_stiffness = 1e7
normal_damping = 10.0
tangential_stiffness = 5e6
tangential_damping = 5.0
bending_stiffness = 1e4
bending_damping = 1.0
break_normal_stretch = 0.1
break_shear = 0.0005
"#;
        let config: BondConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.normal_stiffness, 1e7);
        assert_eq!(config.tangential_stiffness, 5e6);
        assert_eq!(config.bending_stiffness, 1e4);
        assert_eq!(config.break_normal_stretch, Some(0.1));
        assert_eq!(config.break_shear, Some(0.0005));
    }

    #[test]
    fn bond_config_with_file_fields() {
        let toml_str = r#"
normal_stiffness = 1e7
normal_damping = 10.0
file = "data.lammps"
format = "lammps_data"
"#;
        let config: BondConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.file.as_deref(), Some("data.lammps"));
        assert_eq!(config.format.as_deref(), Some("lammps_data"));
    }

    #[test]
    fn bond_config_without_file_fields() {
        let toml_str = r#"
auto_bond = true
normal_stiffness = 1e7
"#;
        let config: BondConfig = toml::from_str(toml_str).unwrap();
        assert!(config.file.is_none());
        assert!(config.auto_bond);
    }
}

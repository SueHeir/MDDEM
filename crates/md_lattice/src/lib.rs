//! FCC lattice initialization with Maxwell-Boltzmann velocity sampling.
//!
//! This crate places atoms on a face-centered cubic (FCC) lattice that fills the
//! simulation domain, then assigns each atom a random velocity drawn from the
//! Maxwell-Boltzmann distribution at a specified temperature. The resulting
//! velocities are corrected to have zero center-of-mass drift and rescaled to
//! match the target temperature exactly.
//!
//! # FCC lattice geometry
//!
//! The FCC unit cell contains **4 atoms** at fractional positions:
//!
//! | Basis atom | Position (fractional) |
//! |:----------:|:---------------------:|
//! | 0          | (0, 0, 0)             |
//! | 1          | (½, ½, 0)             |
//! | 2          | (½, 0, ½)             |
//! | 3          | (0, ½, ½)             |
//!
//! The ideal lattice constant is `a = (4 / ρ)^(1/3)` where `ρ` is the number
//! density. The domain is tiled with `n = floor(L / a)` cells per axis, and the
//! lattice constant is then adjusted to `L / n` so that the lattice fits the
//! domain exactly (with a slightly adjusted density).
//!
//! # Velocity initialization
//!
//! Each velocity component is drawn from `N(0, σ_v)` where `σ_v = √(T / m)`.
//! After sampling:
//! 1. The center-of-mass velocity is subtracted (zero net momentum).
//! 2. Velocities are rescaled so the kinetic temperature matches `T` exactly,
//!    using `T = 2 KE / (3N − 3)`.
//!
//! # TOML configuration
//!
//! ```toml
//! [lattice]
//! style = "fcc"           # Lattice type (only "fcc" supported)
//! density = 0.85          # Number density ρ* (atoms per unit volume)
//! temperature = 0.85      # Initial temperature T* for velocity sampling
//! mass = 1.0              # Default atom mass (LJ reduced units)
//! skin = 1.25             # Neighbor-list skin distance
//!
//! # Optional: multi-type systems
//! # type_fractions = [0.8, 1.0]   # Cumulative fractions → 80% type 0, 20% type 1
//! # type_masses = [1.0, 2.0]      # Per-type masses (overrides `mass`)
//! ```
//!
//! # Plugin usage
//!
//! ```ignore
//! app.add_plugins(LatticePlugin);
//! ```

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;

use mddem_core::{Atom, AtomPlugin, CommResource, Config, Domain, ScheduleSetupSet};

// ── Config ──────────────────────────────────────────────────────────────────

fn default_density() -> f64 {
    0.85
}
fn default_temperature() -> f64 {
    0.85
}
fn default_mass() -> f64 {
    1.0
}
fn default_skin() -> f64 {
    1.25
}

/// TOML `[lattice]` section — controls FCC lattice geometry and velocity initialization.
///
/// All fields have sensible defaults for a standard Lennard-Jones fluid in reduced units.
///
/// # Example
///
/// ```toml
/// [lattice]
/// style = "fcc"
/// density = 0.85
/// temperature = 0.85
/// mass = 1.0
/// skin = 1.25
/// ```
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct LatticeConfig {
    /// Lattice type. Currently only `"fcc"` is supported.
    ///
    /// Default: `"fcc"`
    #[serde(default = "default_style")]
    pub style: String,

    /// Number density ρ* (atoms per unit volume, LJ reduced units).
    ///
    /// Together with the domain size this determines how many unit cells fit
    /// along each axis: `n = floor(L / (4/ρ)^(1/3))`.
    ///
    /// Default: `0.85`
    #[serde(default = "default_density")]
    pub density: f64,

    /// Initial temperature T* for Maxwell-Boltzmann velocity sampling.
    ///
    /// Each velocity component is drawn from `N(0, √(T/m))`, then the ensemble
    /// is corrected to have zero COM drift and rescaled to this exact temperature.
    ///
    /// Default: `0.85`
    #[serde(default = "default_temperature")]
    pub temperature: f64,

    /// Default atom mass (LJ reduced units).
    ///
    /// Overridden on a per-type basis if [`type_masses`](Self::type_masses) is set.
    ///
    /// Default: `1.0`
    #[serde(default = "default_mass")]
    pub mass: f64,

    /// Neighbor-list skin distance.
    ///
    /// Sets the cutoff radius stored per atom for neighbor list construction.
    /// Larger values rebuild the list less often but increase the number of
    /// pair interactions checked.
    ///
    /// Default: `1.25`
    #[serde(default = "default_skin")]
    pub skin: f64,

    /// Cumulative type fractions for multi-type initialization.
    ///
    /// Each entry is a cumulative threshold in `[0, 1]`. Atoms are assigned the
    /// first type whose threshold exceeds their fractional index.
    ///
    /// For example, `[0.8, 1.0]` creates 80% type 0 and 20% type 1.
    /// If absent or empty, all atoms are type 0.
    #[serde(default)]
    pub type_fractions: Option<Vec<f64>>,

    /// Per-type masses. If present, overrides the global [`mass`](Self::mass)
    /// for each atom type. Index corresponds to atom type (0, 1, 2, …).
    #[serde(default)]
    pub type_masses: Option<Vec<f64>>,
}

fn default_style() -> String {
    "fcc".to_string()
}

impl Default for LatticeConfig {
    fn default() -> Self {
        LatticeConfig {
            style: "fcc".to_string(),
            density: 0.85,
            temperature: 0.85,
            mass: 1.0,
            skin: 1.25,
            type_fractions: None,
            type_masses: None,
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Plugin that populates the simulation domain with atoms on an FCC lattice
/// and assigns Maxwell-Boltzmann velocities at the configured temperature.
///
/// Registered systems:
/// - [`fcc_insert`] — places atoms and samples velocities (runs at `Setup`).
/// - [`lattice_set_dt`] — sets the timestep to `0.005` (LJ reduced units, runs at `PostSetup`).
///
/// Both systems run only on the first stage and only on rank 0.
pub struct LatticePlugin;

impl Plugin for LatticePlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[lattice]
style = "fcc"
density = 0.85       # number density rho*
temperature = 0.85   # initial T* for Maxwell-Boltzmann velocities
mass = 1.0           # particle mass
skin = 1.25          # neighbor skin distance"#,
        )
    }

    fn build(&self, app: &mut App) {
        // Ensure Atom + AtomDataRegistry exist (LJ doesn't use DemAtomPlugin)
        if app.get_resource_ref::<Atom>().is_none() {
            app.add_plugins(AtomPlugin);
        }

        Config::load::<LatticeConfig>(app, "lattice");

        app.add_setup_system(
                fcc_insert.run_if(first_stage_only()),
                ScheduleSetupSet::Setup,
            )
            .add_setup_system(
                lattice_set_dt.run_if(first_stage_only()),
                ScheduleSetupSet::PostSetup,
            );
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Places atoms on an FCC lattice and assigns Maxwell-Boltzmann velocities.
///
/// This system:
/// 1. Computes the number of FCC unit cells that fit along each axis.
/// 2. Adjusts the lattice constant so cells tile the domain exactly.
/// 3. Inserts 4 atoms per cell at the FCC basis positions.
/// 4. Draws velocities from `N(0, √(T/m))` per component.
/// 5. Removes center-of-mass drift and rescales to the exact target temperature.
///
/// Only runs on MPI rank 0. Atoms are redistributed to other ranks during
/// the communication phase that follows setup.
///
/// # Panics
///
/// Exits the process if:
/// - The domain is too small to fit even one unit cell along any axis.
/// - The configured temperature is negative.
/// - Any atom mass is zero or negative.
pub fn fcc_insert(
    lattice: Res<LatticeConfig>,
    comm: Res<CommResource>,
    domain: Res<Domain>,
    mut atom: ResMut<Atom>,
) {
    if comm.rank() != 0 {
        return;
    }

    let rho = lattice.density;
    let temp = lattice.temperature;
    let mass = lattice.mass;

    // ── Compute lattice geometry ────────────────────────────────────────
    // FCC has 4 atoms per unit cell. From number density ρ = 4/a³,
    // the ideal lattice constant is a = (4/ρ)^(1/3).
    let a_ideal = (4.0 / rho).cbrt();

    let lx = domain.size[0];
    let ly = domain.size[1];
    let lz = domain.size[2];

    // Number of unit cells that fit along each axis (rounded down)
    let nx = (lx / a_ideal).floor() as usize;
    let ny = (ly / a_ideal).floor() as usize;
    let nz = (lz / a_ideal).floor() as usize;

    if nx == 0 || ny == 0 || nz == 0 {
        eprintln!(
            "ERROR: Domain too small for FCC lattice: L=({},{},{}), a_ideal={:.4}. \
             Increase domain size or decrease density.",
            lx, ly, lz, a_ideal
        );
        std::process::exit(1);
    }

    // Stretch the lattice constant slightly so unit cells tile the domain
    // exactly, avoiding a gap at the boundary.
    let ax = lx / nx as f64;
    let ay = ly / ny as f64;
    let az = lz / nz as f64;

    // FCC basis: 4 atoms at the corners and face-centers of the cubic unit cell.
    //   (0,0,0)   — corner atom
    //   (½,½,0)   — face center on the xy-face
    //   (½,0,½)   — face center on the xz-face
    //   (0,½,½)   — face center on the yz-face
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];

    let x0 = domain.boundaries_low[0];
    let y0 = domain.boundaries_low[1];
    let z0 = domain.boundaries_low[2];

    // ── Multi-type setup ────────────────────────────────────────────────
    // Build cumulative fraction thresholds for deterministic type assignment.
    let fractions: Vec<f64> = match &lattice.type_fractions {
        Some(f) if !f.is_empty() => f.clone(),
        _ => vec![1.0], // single-type: all atoms are type 0
    };

    // Build per-type mass table (falls back to the global `mass` for all types).
    let type_masses: Vec<f64> = match &lattice.type_masses {
        Some(m) if !m.is_empty() => m.clone(),
        _ => vec![mass; fractions.len()],
    };

    let mut max_tag = atom.get_max_tag();
    let start_idx = atom.len();

    // ── Precompute type assignments ─────────────────────────────────────
    // Each atom's type is chosen by mapping its sequential index to a
    // fraction in [0, 1) and finding the first cumulative threshold that
    // exceeds it. This gives a deterministic, reproducible assignment.
    let total_to_insert = 4 * nx * ny * nz;
    let mut type_counts = vec![0usize; fractions.len()];
    let mut type_assignments = Vec::with_capacity(total_to_insert);
    for idx in 0..total_to_insert {
        let frac = (idx as f64 + 0.5) / total_to_insert as f64;
        let mut atype = 0u32;
        for (t, &threshold) in fractions.iter().enumerate() {
            if frac < threshold {
                atype = t as u32;
                break;
            }
            // Past the last threshold — assign the last type
            atype = t as u32;
        }
        type_assignments.push(atype);
        type_counts[atype as usize] += 1;
    }

    // ── Insert atoms ────────────────────────────────────────────────────
    let mut atom_idx = 0usize;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                for b in &basis {
                    let x = x0 + (ix as f64 + b[0]) * ax;
                    let y = y0 + (iy as f64 + b[1]) * ay;
                    let z = z0 + (iz as f64 + b[2]) * az;

                    let atype = type_assignments[atom_idx];
                    let m = type_masses.get(atype as usize).copied().unwrap_or(mass);

                    atom.tag.push(max_tag);
                    atom.atom_type.push(atype);
                    atom.origin_index.push(0);
                    atom.pos.push([x, y, z]);
                    atom.vel.push([0.0; 3]);
                    atom.force.push([0.0; 3]);
                    atom.mass.push(m);
                    atom.inv_mass.push(1.0 / m);
                    atom.cutoff_radius.push(lattice.skin);
                    atom.is_ghost.push(false);
                    max_tag += 1;
                    atom.nlocal += 1;
                    atom.natoms += 1;
                    atom_idx += 1;
                }
            }
        }
    }

    let n_inserted = atom.len() - start_idx;

    // ── Maxwell-Boltzmann velocity sampling ─────────────────────────────
    // Each velocity component v_i is drawn from a Gaussian with zero mean
    // and standard deviation σ_v = √(T/m), which is the Maxwell-Boltzmann
    // distribution for a single component in reduced units.
    if temp < 0.0 {
        eprintln!("ERROR: lattice temperature must be non-negative, got {}", temp);
        std::process::exit(1);
    }
    let mut rng = rand::rng();

    for i in start_idx..atom.len() {
        let m = atom.mass[i];
        if m <= 0.0 {
            eprintln!("ERROR: lattice mass must be positive, got {}", m);
            std::process::exit(1);
        }
        let sigma_v = (temp / m).sqrt();
        let normal = Normal::new(0.0, sigma_v)
            .expect("failed to create Normal distribution: sigma_v must be finite and non-negative");
        atom.vel[i][0] = normal.sample(&mut rng);
        atom.vel[i][1] = normal.sample(&mut rng);
        atom.vel[i][2] = normal.sample(&mut rng);
    }

    // ── Remove center-of-mass drift ─────────────────────────────────────
    // Subtract the mean velocity so the system has zero net momentum,
    // preventing the entire lattice from translating.
    let n = n_inserted as f64;
    let mut vx_sum = 0.0;
    let mut vy_sum = 0.0;
    let mut vz_sum = 0.0;
    for i in start_idx..atom.len() {
        vx_sum += atom.vel[i][0];
        vy_sum += atom.vel[i][1];
        vz_sum += atom.vel[i][2];
    }
    let vx_avg = vx_sum / n;
    let vy_avg = vy_sum / n;
    let vz_avg = vz_sum / n;
    for i in start_idx..atom.len() {
        atom.vel[i][0] -= vx_avg;
        atom.vel[i][1] -= vy_avg;
        atom.vel[i][2] -= vz_avg;
    }

    // ── Rescale to exact target temperature ─────────────────────────────
    // After COM removal, the instantaneous temperature from the sampled
    // velocities won't match T exactly. Rescale velocities so that
    //   T_actual = 2·KE / N_dof = T_target
    // where N_dof = 3N − 3 (3 translational degrees of freedom removed).
    let ndof = 3.0 * n - 3.0;
    if ndof > 0.0 {
        let mut ke = 0.0;
        for i in start_idx..atom.len() {
            ke += atom.mass[i]
                * (atom.vel[i][0].powi(2) + atom.vel[i][1].powi(2) + atom.vel[i][2].powi(2));
        }
        ke *= 0.5;
        let current_temp = 2.0 * ke / ndof;
        if current_temp > 1e-20 {
            let scale = (temp / current_temp).sqrt();
            for i in start_idx..atom.len() {
                atom.vel[i][0] *= scale;
                atom.vel[i][1] *= scale;
                atom.vel[i][2] *= scale;
            }
        }
    }

    println!(
        "Lattice: inserted {} atoms (FCC {}x{}x{}), a=({:.4},{:.4},{:.4}), rho_actual={:.4}",
        n_inserted,
        nx,
        ny,
        nz,
        ax,
        ay,
        az,
        n_inserted as f64 / domain.volume
    );
    if fractions.len() > 1 {
        for (t, &count) in type_counts.iter().enumerate() {
            println!(
                "  Type {}: {} atoms ({:.1}%)",
                t,
                count,
                100.0 * count as f64 / n_inserted as f64
            );
        }
    }
}

/// Sets the integration timestep to `0.005` (standard for LJ reduced units).
///
/// This system runs once during setup on rank 0. The value `dt = 0.005` is a
/// widely used default for Lennard-Jones simulations in reduced units and
/// provides a good balance between accuracy and performance.
pub fn lattice_set_dt(
    comm: Res<CommResource>,
    mut atoms: ResMut<Atom>,
) {
    let dt = 0.005; // Standard LJ reduced-unit timestep
    atoms.dt = dt;

    if comm.rank() == 0 {
        println!("Using dt={} (LJ reduced units)", dt);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::{CommBackend, SingleProcessComm};

    fn make_test_app(lx: f64, rho: f64) -> App {
        let mut app = App::new();

        let lattice = LatticeConfig {
            style: "fcc".to_string(),
            density: rho,
            temperature: 0.85,
            mass: 1.0,
            skin: 1.25,
            type_fractions: None,
            type_masses: None,
        };
        app.add_resource(lattice);

        let mut comm = SingleProcessComm::new();
        comm.set_processor_grid([1, 1, 1], [0, 0, 0]);
        app.add_resource(mddem_core::CommResource(Box::new(comm)));

        let mut domain = Domain::new();
        domain.boundaries_low = [0.0, 0.0, 0.0];
        domain.boundaries_high = [lx, lx, lx];
        domain.size = [lx, lx, lx];
        domain.volume = lx * lx * lx;
        app.add_resource(domain);

        app.add_resource(Atom::new());

        let mut sm = SchedulerManager::new();
        sm.index = 0;
        app.add_resource(sm);

        app.add_setup_system(fcc_insert, ScheduleSetupSet::Setup);
        app.organize_systems();
        app
    }

    #[test]
    fn fcc_correct_atom_count() {
        let rho: f64 = 0.85;
        let a = (4.0_f64 / rho).cbrt();
        // Use exact multiple of lattice constant so floor doesn't reduce cell count
        let ncells = 6_usize;
        let lx = ncells as f64 * a;
        let mut app = make_test_app(lx, rho);
        app.setup();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        let expected = 4 * ncells * ncells * ncells;
        assert_eq!(
            atom.len(),
            expected,
            "Expected {} atoms, got {}",
            expected,
            atom.len()
        );
    }

    #[test]
    fn fcc_density_matches() {
        let rho: f64 = 0.85;
        let a = (4.0_f64 / rho).cbrt();
        let ncells = 6_usize;
        let lx = ncells as f64 * a;
        let mut app = make_test_app(lx, rho);
        app.setup();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        let domain = app.get_resource_ref::<Domain>().unwrap();
        let actual_rho = atom.len() as f64 / domain.volume;
        assert!(
            (actual_rho - rho).abs() < 0.01,
            "Density mismatch: expected ~{}, got {}",
            rho,
            actual_rho
        );
    }

    #[test]
    fn fcc_zero_com_velocity() {
        let rho: f64 = 0.85;
        let a = (4.0_f64 / rho).cbrt();
        let ncells = 4_usize;
        let lx = ncells as f64 * a;
        let mut app = make_test_app(lx, rho);
        app.setup();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        let n = atom.len() as f64;
        let vx_com: f64 = atom.vel.iter().map(|v| v[0]).sum::<f64>() / n;
        let vy_com: f64 = atom.vel.iter().map(|v| v[1]).sum::<f64>() / n;
        let vz_com: f64 = atom.vel.iter().map(|v| v[2]).sum::<f64>() / n;
        assert!(
            vx_com.abs() < 1e-12,
            "COM velocity x should be ~0: {}",
            vx_com
        );
        assert!(
            vy_com.abs() < 1e-12,
            "COM velocity y should be ~0: {}",
            vy_com
        );
        assert!(
            vz_com.abs() < 1e-12,
            "COM velocity z should be ~0: {}",
            vz_com
        );
    }
}

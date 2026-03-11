//! FCC lattice initialization with Maxwell-Boltzmann velocity sampling.

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
#[cfg(test)]
use nalgebra::Vector3;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;

use mddem_core::{Atom, AtomPlugin, CommResource, Config, Domain};

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

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[lattice]` — lattice initialization settings.
pub struct LatticeConfig {
    /// Lattice type (currently only `"fcc"`).
    #[serde(default = "default_style")]
    pub style: String,
    /// Number density (atoms per unit volume).
    #[serde(default = "default_density")]
    pub density: f64,
    /// Initial temperature for Maxwell-Boltzmann velocity sampling.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Atom mass.
    #[serde(default = "default_mass")]
    pub mass: f64,
    /// Interaction skin distance (neighbor list cutoff parameter).
    #[serde(default = "default_skin")]
    pub skin: f64,
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
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Initializes atoms on an FCC lattice with Maxwell-Boltzmann velocities at setup.
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

        app.add_setup_system(fcc_insert, ScheduleSetupSet::Setup)
            .add_setup_system(lattice_set_dt, ScheduleSetupSet::PostSetup);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// FCC lattice insertion with Maxwell-Boltzmann velocities.
/// Only inserts on rank 0, only on stage 0.
pub fn fcc_insert(
    lattice: Res<LatticeConfig>,
    comm: Res<CommResource>,
    domain: Res<Domain>,
    mut atom: ResMut<Atom>,
    scheduler_manager: Res<SchedulerManager>,
) {
    if scheduler_manager.index != 0 {
        return;
    }
    if comm.rank() != 0 {
        return;
    }

    let rho = lattice.density;
    let temp = lattice.temperature;
    let mass = lattice.mass;

    // FCC: 4 atoms per unit cell, a = (4/rho)^(1/3)
    let a_ideal = (4.0 / rho).cbrt();

    let lx = domain.size.x;
    let ly = domain.size.y;
    let lz = domain.size.z;

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

    // Adjusted lattice constants to fit domain exactly
    let ax = lx / nx as f64;
    let ay = ly / ny as f64;
    let az = lz / nz as f64;

    // FCC basis positions (fractional coordinates within unit cell)
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];

    let x0 = domain.boundaries_low.x;
    let y0 = domain.boundaries_low.y;
    let z0 = domain.boundaries_low.z;

    let mut max_tag = atom.get_max_tag();
    let start_idx = atom.len();

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                for b in &basis {
                    let x = x0 + (ix as f64 + b[0]) * ax;
                    let y = y0 + (iy as f64 + b[1]) * ay;
                    let z = z0 + (iz as f64 + b[2]) * az;

                    atom.tag.push(max_tag);
                    atom.atom_type.push(0);
                    atom.origin_index.push(0);
                    atom.pos.push([x, y, z]);
                    atom.vel.push([0.0; 3]);
                    atom.force.push([0.0; 3]);
                    atom.mass.push(mass);
                    atom.inv_mass.push(1.0 / mass);
                    atom.skin.push(lattice.skin);
                    atom.is_ghost.push(false);
                    atom.is_collision.push(false);
                    max_tag += 1;
                    atom.nlocal += 1;
                    atom.natoms += 1;
                }
            }
        }
    }

    let n_inserted = atom.len() - start_idx;

    // Assign Maxwell-Boltzmann velocities
    if mass <= 0.0 {
        eprintln!("ERROR: lattice mass must be positive, got {}", mass);
        std::process::exit(1);
    }
    if temp < 0.0 {
        eprintln!("ERROR: lattice temperature must be non-negative, got {}", temp);
        std::process::exit(1);
    }
    let sigma_v = (temp / mass).sqrt();
    let normal = Normal::new(0.0, sigma_v).unwrap();
    let mut rng = rand::rng();

    for i in start_idx..atom.len() {
        atom.vel[i][0] = normal.sample(&mut rng);
        atom.vel[i][1] = normal.sample(&mut rng);
        atom.vel[i][2] = normal.sample(&mut rng);
    }

    // Remove COM drift
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

    // Rescale to exact target temperature
    // T = (2*KE)/(ndof) where KE = 0.5*m*v^2, ndof = 3N-3
    let ndof = 3.0 * n - 3.0;
    if ndof > 0.0 {
        let mut ke = 0.0;
        for i in start_idx..atom.len() {
            ke += mass
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
}

/// Set timestep for LJ reduced units (default dt=0.005)
pub fn lattice_set_dt(
    comm: Res<CommResource>,
    mut atoms: ResMut<Atom>,
    scheduler_manager: Res<SchedulerManager>,
) {
    if scheduler_manager.index != 0 {
        return;
    }

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
        };
        app.add_resource(lattice);

        let mut comm = SingleProcessComm::new();
        comm.set_processor_grid(Vector3::new(1, 1, 1), Vector3::new(0, 0, 0));
        app.add_resource(mddem_core::CommResource(Box::new(comm)));

        let mut domain = Domain::new();
        domain.boundaries_low = Vector3::new(0.0, 0.0, 0.0);
        domain.boundaries_high = Vector3::new(lx, lx, lx);
        domain.size = Vector3::new(lx, lx, lx);
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

//! Rotating drum angle of repose benchmark.
//!
//! Validates the dynamic angle of repose in a 2D rotating drum against
//! expected trends: angle of repose increases with friction coefficient.
//!
//! The drum wall is modeled as a ring of frozen particles that are rotated
//! kinematically. This gives full Hertz-Mindlin tangential friction between
//! the drum wall and mobile particles.
//!
//! ```bash
//! cargo run --release --example bench_rotating_drum --no-default-features \
//!     -- examples/bench_rotating_drum/config.toml
//! ```

use mddem::dem_atom::DemAtom;
use mddem::prelude::*;
use std::f64::consts::PI;
use std::fs::{self, OpenOptions};
use std::io::Write;

// ── Drum constants ───────────────────────────────────────────────────────────

const DRUM_RADIUS: f64 = 0.05; // m
const DRUM_CENTER_X: f64 = 0.05; // m
const DRUM_CENTER_Z: f64 = 0.05; // m
const DRUM_OMEGA: f64 = 1.5; // rad/s  (Fr = omega^2 R/g ≈ 0.011)
const SETTLING_STEPS: u64 = 80_000; // 4.0 s at dt=5e-5
const MEASURE_INTERVAL: u64 = 500; // steps between angle measurements
const WALL_PARTICLE_RADIUS: f64 = 0.002; // same as mobile particles
const WALL_PARTICLE_TYPE: u32 = 1; // type index for wall particles
const MOBILE_PARTICLE_TYPE: u32 = 0; // type index for mobile particles
const OUTPUT_DIR: &str = "examples/bench_rotating_drum/output";

// ── Drum plugin ──────────────────────────────────────────────────────────────

/// Plugin that creates a ring of frozen wall particles and rotates them.
struct DrumPlugin;

impl Plugin for DrumPlugin {
    fn build(&self, app: &mut App) {
        // Add systems for drum wall rotation, force zeroing, and measurement
        app.add_update_system(rotate_drum_wall, ScheduleSet::PreInitialIntegration);
        app.add_update_system(zero_wall_particle_forces, ScheduleSet::PostForce);
        app.add_update_system(measure_surface_angle, ScheduleSet::PostFinalIntegration);
    }
}

// ── Wall particle creation ───────────────────────────────────────────────────

/// Plugin that creates a ring of wall particles after DEM atom setup.
/// Must run after GranularDefaultPlugins to access Atom and DemAtom.
struct WallParticlePlugin;

impl Plugin for WallParticlePlugin {
    fn build(&self, app: &mut App) {
        app.add_setup_system(create_wall_particles, ScheduleSetupSet::PostSetup);
    }
}

/// Create a ring of particles at the drum periphery.
/// These will be rotated kinematically each timestep.
fn create_wall_particles(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    let mut dem = registry.expect_mut::<DemAtom>("create_wall_particles");

    let r = WALL_PARTICLE_RADIUS;
    let density = 2500.0;

    // Compute number of wall particles to cover the circumference
    let circumference = 2.0 * PI * DRUM_RADIUS;
    let n_wall = (circumference / (2.0 * r)).ceil() as usize; // touching, no overlap
    let mass = density * 4.0 / 3.0 * PI * r * r * r;

    // Starting tag after any existing particles
    let start_tag = atoms.natoms as u32;

    // Update ntypes to include wall particle type
    if atoms.ntypes < 2 {
        atoms.ntypes = 2;
    }

    for k in 0..n_wall {
        let angle = 2.0 * PI * (k as f64) / (n_wall as f64);
        let x = DRUM_CENTER_X + DRUM_RADIUS * angle.cos();
        let z = DRUM_CENTER_Z + DRUM_RADIUS * angle.sin();
        let y = 0.0025; // center of thin slab

        atoms.natoms += 1;
        atoms.nlocal += 1;
        atoms.tag.push(start_tag + k as u32);
        atoms.origin_index.push(0);
        atoms.cutoff_radius.push(r);
        atoms.image.push([0, 0, 0]);
        atoms.is_ghost.push(false);
        atoms.pos.push([x, y, z]);
        atoms.vel.push([0.0; 3]); // velocity set by rotate system
        atoms.force.push([0.0; 3]);
        atoms.mass.push(mass);
        atoms.inv_mass.push(1.0 / mass);
        atoms.atom_type.push(WALL_PARTICLE_TYPE);
        dem.radius.push(r);
        dem.density.push(density);
        dem.inv_inertia.push(1.0 / (0.4 * mass * r * r));
        dem.quaternion.push([1.0, 0.0, 0.0, 0.0]);
        dem.omega.push([0.0; 3]);
        dem.ang_mom.push([0.0; 3]);
        dem.torque.push([0.0; 3]);
        dem.body_id.push(0.0);
    }
}

// ── Drum rotation system ─────────────────────────────────────────────────────

/// Rotates wall particles (type 1) around the drum center.
///
/// Before settling completes: wall particles are static.
/// After settling: wall particles follow circular orbits at angular velocity omega.
///
/// Each wall particle's initial angle is determined from its position.
fn rotate_drum_wall(mut atoms: ResMut<Atom>, run_state: Res<RunState>) {
    let step = run_state.total_cycle as u64;
    let dt = atoms.dt;

    if step < SETTLING_STEPS {
        // During settling, wall particles are static (velocity = 0)
        return;
    }

    let _t_rot = (step - SETTLING_STEPS) as f64 * dt;
    let nlocal = atoms.nlocal as usize;

    for i in 0..nlocal {
        if atoms.atom_type[i] != WALL_PARTICLE_TYPE {
            continue;
        }

        // Wall particle's initial angle is stored via its tag.
        // Recompute from current position (which we set each step).
        let dx = atoms.pos[i][0] - DRUM_CENTER_X;
        let dz = atoms.pos[i][2] - DRUM_CENTER_Z;
        let current_angle = dz.atan2(dx);

        // Advance by omega * dt (incremental rotation per step)
        let angle = current_angle + DRUM_OMEGA * dt;

        // Set position
        atoms.pos[i][0] = DRUM_CENTER_X + DRUM_RADIUS * angle.cos();
        atoms.pos[i][2] = DRUM_CENTER_Z + DRUM_RADIUS * angle.sin();

        // Set tangential velocity (perpendicular to radius, CCW)
        atoms.vel[i][0] = -DRUM_RADIUS * DRUM_OMEGA * angle.sin();
        atoms.vel[i][1] = 0.0;
        atoms.vel[i][2] = DRUM_RADIUS * DRUM_OMEGA * angle.cos();
    }
}

/// Zero forces on wall particles so velocity Verlet doesn't change their velocity.
fn zero_wall_particle_forces(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    let mut dem = registry.expect_mut::<DemAtom>("zero_wall_particle_forces");
    let nlocal = atoms.nlocal as usize;

    for i in 0..nlocal {
        if atoms.atom_type[i] != WALL_PARTICLE_TYPE {
            continue;
        }
        atoms.force[i] = [0.0; 3];
        dem.torque[i] = [0.0; 3];
    }
}

// ── Surface angle measurement ────────────────────────────────────────────────

/// Measures the dynamic angle of repose by fitting a line through the free
/// surface of mobile particles.
///
/// Algorithm:
/// 1. Select only mobile particles (type 0)
/// 2. Bin by x-coordinate
/// 3. For each bin, find the highest z (free surface)
/// 4. Fit a line through (x, z_max) using least squares
/// 5. Angle = atan(|slope|) in degrees
fn measure_surface_angle(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    registry: Res<AtomDataRegistry>,
) {
    let step = run_state.total_cycle as u64;

    // Only measure during rotation phase, at the specified interval
    if step < SETTLING_STEPS || step % MEASURE_INTERVAL != 0 {
        return;
    }

    // Wait 1.5 full rotations for transient to pass
    let rotation_steps = step - SETTLING_STEPS;
    let t_rot = rotation_steps as f64 * atoms.dt;
    let period = 2.0 * PI / DRUM_OMEGA;
    if t_rot < 1.5 * period {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    let _dem = registry.get::<DemAtom>().unwrap();

    // Collect mobile particle positions
    let mut positions: Vec<(f64, f64)> = Vec::new();
    for i in 0..nlocal {
        if atoms.atom_type[i] != MOBILE_PARTICLE_TYPE {
            continue;
        }
        positions.push((atoms.pos[i][0], atoms.pos[i][2]));
    }

    if positions.len() < 20 {
        return;
    }

    // Bin particles by x to find the free surface
    let n_bins: usize = 12;
    let bin_lo = DRUM_CENTER_X - 0.6 * DRUM_RADIUS;
    let bin_hi = DRUM_CENTER_X + 0.6 * DRUM_RADIUS;
    let bin_width = (bin_hi - bin_lo) / n_bins as f64;

    let mut bin_z_max = vec![f64::NEG_INFINITY; n_bins];
    let mut bin_count = vec![0u32; n_bins];

    for &(x, z) in &positions {
        if x < bin_lo || x >= bin_hi {
            continue;
        }
        let bin_idx = ((x - bin_lo) / bin_width) as usize;
        let bin_idx = bin_idx.min(n_bins - 1);
        bin_count[bin_idx] += 1;
        if z > bin_z_max[bin_idx] {
            bin_z_max[bin_idx] = z;
        }
    }

    // Collect valid surface points
    let min_per_bin = 2u32;
    let mut surface_points: Vec<(f64, f64)> = Vec::new();
    for b in 0..n_bins {
        if bin_count[b] >= min_per_bin && bin_z_max[b] > f64::NEG_INFINITY {
            let x_center = bin_lo + (b as f64 + 0.5) * bin_width;
            surface_points.push((x_center, bin_z_max[b]));
        }
    }

    if surface_points.len() < 4 {
        return;
    }

    // Least squares line fit: z = a * x + b
    let n_pts = surface_points.len() as f64;
    let sum_x: f64 = surface_points.iter().map(|(x, _)| x).sum();
    let sum_z: f64 = surface_points.iter().map(|(_, z)| z).sum();
    let sum_xz: f64 = surface_points.iter().map(|(x, z)| x * z).sum();
    let sum_xx: f64 = surface_points.iter().map(|(x, _)| x * x).sum();

    let denom = n_pts * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return;
    }
    let slope = (n_pts * sum_xz - sum_x * sum_z) / denom;
    let angle_deg = slope.atan().abs() * 180.0 / PI;

    // Only rank 0 writes
    if comm.rank() != 0 {
        return;
    }

    let _ = fs::create_dir_all(OUTPUT_DIR);

    let filepath = format!("{}/surface_angle.txt", OUTPUT_DIR);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&filepath)
        .unwrap();

    // Write header if file is empty
    if file.metadata().unwrap().len() == 0 {
        writeln!(file, "step time angle_deg").unwrap();
    }

    let time = step as f64 * atoms.dt;
    writeln!(file, "{} {:.6e} {:.4}", step, time, angle_deg).unwrap();
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(GravityPlugin)
        .add_plugins(WallParticlePlugin)
        .add_plugins(DrumPlugin);

    app.start();
}

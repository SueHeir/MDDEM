//! Triaxial compression benchmark: validates DEM against Mohr-Coulomb failure theory.
//!
//! Stages: insert → relax → compress
//! - Insert: random particles settle under gravity in a walled box
//! - Relax: wait for kinetic energy to decay
//! - Compress: top wall moves down at constant velocity; lateral walls use
//!   servo control to maintain confining pressure; gravity set to zero for
//!   uniform stress distribution
//!
//! Run at different confining pressures via separate config files:
//! ```bash
//! cargo run --release --example bench_triaxial --no-default-features -- examples/bench_triaxial/config_10kPa.toml
//! ```

use mddem::prelude::*;
use std::fs::{self, File, OpenOptions};
use std::io::Write;

/// Axial compression velocity [m/s].  Chosen for quasi-static loading
/// (inertial number I ≪ 0.01 at all confining pressures tested).
const COMPRESS_VEL: f64 = 5e-4;

#[derive(Clone, Debug, PartialEq, Default, StageEnum)]
enum Phase {
    #[default]
    #[stage("insert")]
    Insert,
    #[stage("relax")]
    Relax,
    #[stage("compress")]
    Compress,
}

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(GravityPlugin)
        .add_plugins(WallPlugin)
        .add_plugins(ContactAnalysisPlugin)
        .add_plugins(StatesPlugin {
            initial: Phase::Insert,
        })
        .add_plugins(StageAdvancePlugin::<Phase>::new());

    // Stage-transition systems
    app.add_update_system(
        check_insert_settled.run_if(in_state(Phase::Insert)),
        ScheduleSet::PostFinalIntegration,
    );
    app.add_update_system(
        check_relaxed.run_if(in_state(Phase::Relax)),
        ScheduleSet::PostFinalIntegration,
    );

    // Compression systems (compress stage only)
    app.add_update_system(
        apply_compression.run_if(in_state(Phase::Compress)),
        ScheduleSet::PreInitialIntegration,
    );
    app.add_update_system(
        output_stress.run_if(in_state(Phase::Compress)),
        ScheduleSet::PostFinalIntegration,
    );

    app.start();
}

// ── Stage transition checks ─────────────────────────────────────────────────

/// Advance to relax when kinetic energy drops below threshold.
fn check_insert_settled(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    mut next_state: ResMut<NextState<Phase>>,
) {
    let step = run_state.total_cycle;
    if step < 1000 || step % 100 != 0 {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    let local_ke: f64 = (0..nlocal)
        .map(|i| {
            let v = atoms.vel[i];
            0.5 * atoms.mass[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        })
        .sum();
    let global_ke = comm.all_reduce_sum_f64(local_ke);

    if global_ke < 1e-5 {
        next_state.set(Phase::Relax);
        if comm.rank() == 0 {
            println!(
                "Step {}: KE = {:.3e} J — particles settled, advancing to relax",
                step, global_ke
            );
        }
    }
}

/// Advance to compress when kinetic energy is sufficiently small.
fn check_relaxed(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    mut next_state: ResMut<NextState<Phase>>,
) {
    let step = run_state.total_cycle;
    if step % 100 != 0 {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    let local_ke: f64 = (0..nlocal)
        .map(|i| {
            let v = atoms.vel[i];
            0.5 * atoms.mass[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        })
        .sum();
    let global_ke = comm.all_reduce_sum_f64(local_ke);

    if global_ke < 1e-7 {
        next_state.set(Phase::Compress);
        if comm.rank() == 0 {
            println!(
                "Step {}: KE = {:.3e} J — fully relaxed, advancing to compress",
                step, global_ke
            );
        }
    }
}

// ── Compression systems ──────────────────────────────────────────────────────

/// Move the top wall (wall index 5) downward at constant velocity.
///
/// On the first call, snaps the top wall to just above the highest particle
/// so compression begins immediately rather than wasting steps traversing
/// empty space.
fn apply_compression(
    mut walls: ResMut<Walls>,
    atoms: Res<Atom>,
    comm: Res<CommResource>,
    mut initialized: Local<bool>,
) {
    let dt = atoms.dt;

    if !*initialized {
        *initialized = true;
        // Snap top wall to just above the particle bed
        let nlocal = atoms.nlocal as usize;
        let mut max_z: f64 = 0.0;
        for i in 0..nlocal {
            let z = atoms.pos[i][2];
            if z > max_z {
                max_z = z;
            }
        }
        // Use -min(-x) trick since there's no all_reduce_max
        let max_z = -comm.all_reduce_min_f64(-max_z);
        let top_wall = &mut walls.planes[5];
        top_wall.point_z = max_z + 0.0015; // ~1.5 × radius gap
        if comm.rank() == 0 {
            println!(
                "Compression: snapped top wall to z = {:.6} m",
                top_wall.point_z
            );
        }
    }

    // Constant-velocity axial compression
    let top_wall = &mut walls.planes[5];
    top_wall.point_z -= COMPRESS_VEL * dt;
    top_wall.velocity = [0.0, 0.0, -COMPRESS_VEL];
}

/// Compute principal stresses from wall force accumulators and write to CSV.
///
/// Wall layout (must match config):
///   0: x = x_lo, normal +x  (servo, left)
///   1: x = x_hi, normal −x  (servo, right)
///   2: y = y_lo, normal +y  (servo, front)
///   3: y = y_hi, normal −y  (servo, back)
///   4: z = z_lo, normal +z  (static floor)
///   5: z = z_hi, normal −z  (static top, moved by apply_compression)
fn output_stress(
    walls: Res<Walls>,
    atoms: Res<Atom>,
    domain: Res<Domain>,
    run_state: Res<RunState>,
    thermo: Res<Thermo>,
    input: Res<Input>,
    comm: Res<CommResource>,
    mut file_created: Local<bool>,
    mut initial_height: Local<f64>,
) {
    if comm.rank() != 0 {
        return;
    }
    if !run_state.total_cycle.is_multiple_of(thermo.interval) {
        return;
    }

    let output_dir = input.output_dir.as_deref().unwrap_or(".");
    let path = format!("{}/triaxial_stress.csv", output_dir);

    // Create file with header on first call
    if !*file_created {
        *file_created = true;
        fs::create_dir_all(output_dir).ok();
        let mut f = File::create(&path).expect("cannot create stress CSV");
        writeln!(f, "step,time,strain,sigma_1,sigma_3,q,p")
            .expect("cannot write CSV header");
        // Record initial sample height for strain computation
        let floor_z = walls.planes[4].point_z;
        let top_z = walls.planes[5].point_z;
        *initial_height = top_z - floor_z;
    }

    let step = run_state.total_cycle;
    let time = step as f64 * atoms.dt;

    let lx = domain.size[0];
    let ly = domain.size[1];
    let floor_z = walls.planes[4].point_z;
    let top_z = walls.planes[5].point_z;
    let h = top_z - floor_z;

    // Axial strain (positive = compression)
    let strain = if *initial_height > 0.0 {
        (*initial_height - h) / *initial_height
    } else {
        0.0
    };

    // σ₁ = axial stress = top wall force / cross-section area
    let area_top = lx * ly;
    let sigma_1 = walls.planes[5].force_accumulator / area_top;

    // σ₃ = confining stress = average of lateral wall stresses
    let f_x_avg =
        (walls.planes[0].force_accumulator + walls.planes[1].force_accumulator) / 2.0;
    let f_y_avg =
        (walls.planes[2].force_accumulator + walls.planes[3].force_accumulator) / 2.0;
    let area_x = ly * h; // yz-face area
    let area_y = lx * h; // xz-face area
    let sigma_3 = if h > 0.0 {
        (f_x_avg / area_x + f_y_avg / area_y) / 2.0
    } else {
        0.0
    };

    // Deviatoric stress and mean stress
    let q = sigma_1 - sigma_3;
    let p = (sigma_1 + 2.0 * sigma_3) / 3.0;

    let mut f = OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("cannot open stress CSV");
    writeln!(
        f,
        "{},{:.10e},{:.10e},{:.6e},{:.6e},{:.6e},{:.6e}",
        step, time, strain, sigma_1, sigma_3, q, p
    )
    .expect("cannot write stress data");
}

//! Triaxial compression benchmark: validates DEM against Mohr-Coulomb failure theory.
//!
//! Stages: insert → relax → compress
//! - Insert: random particles settle under gravity in a walled box (servos disabled)
//! - Relax: wait for kinetic energy to decay (servos disabled)
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
const COMPRESS_VEL: f64 = 1e-3;

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
        .add_plugins(StatesPlugin {
            initial: Phase::Insert,
        })
        .add_plugins(StageAdvancePlugin::<Phase>::new());

    // Disable servo walls during insert/relax so particles can settle under gravity.
    app.add_update_system(
        disable_servos_during_settling.run_if(in_stage("insert")),
        ScheduleSet::PreInitialIntegration,
    );

    // Compression systems (compress stage only)
    app.add_update_system(
        apply_compression.run_if(in_stage("compress")),
        ScheduleSet::PreInitialIntegration,
    );
    app.add_update_system(
        output_stress.run_if(in_stage("compress")),
        ScheduleSet::PostFinalIntegration,
    );

    app.start();
}

// ── Servo management ───────────────────────────────────────────────────────

/// On the first step, disable servos on walls 0-3 so particles settle under pure gravity.
fn disable_servos_during_settling(
    mut walls: ResMut<Walls>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    *done = true;
    for i in 0..4 {
        walls.planes[i].motion = WallMotion::Static;
        walls.planes[i].velocity = [0.0; 3];
    }
}

// ── Compression systems ──────────────────────────────────────────────────────

/// Stored confining pressure [Pa] and servo parameters, computed at initialization.
struct ConfiningState {
    sigma_3: f64,
    max_velocity: f64,
    gain: f64,
}

/// Move the top wall (wall index 5) downward at constant velocity.
/// Update servo target forces based on current bed height to maintain constant confining pressure.
///
/// On the first call, snaps the top wall to just above the highest particle
/// and re-enables servo walls for lateral confinement.
fn apply_compression(
    mut walls: ResMut<Walls>,
    atoms: Res<Atom>,
    config: Res<Config>,
    comm: Res<CommResource>,
    mut state: Local<Option<ConfiningState>>,
) {
    let dt = atoms.dt;

    if state.is_none() {
        // Snap top wall to just above the particle bed
        let nlocal = atoms.nlocal as usize;
        let mut max_z: f64 = 0.0;
        for i in 0..nlocal {
            let z = atoms.pos[i][2];
            if z > max_z {
                max_z = z;
            }
        }
        let max_z = -comm.all_reduce_min_f64(-max_z);
        let top_wall = &mut walls.planes[5];
        top_wall.point_z = max_z + 0.00105; // radius + 0.05mm surface gap
        if comm.rank() == 0 {
            println!(
                "Compression: snapped top wall to z = {:.6} m",
                top_wall.point_z
            );
        }

        // Read confining pressure from [triaxial] config section
        let sigma_3 = config.table
            .get("triaxial")
            .and_then(|t| t.get("confining_pressure"))
            .and_then(|v| v.as_float())
            .expect("[triaxial] confining_pressure must be set in config");

        // Read servo gain/max_velocity from the first servo wall definition
        let wall_defs: Vec<WallDef> = config.section("wall");
        let srv = wall_defs[0].servo.as_ref().expect("wall 0 must have servo config");

        let h = top_wall.point_z - walls.planes[4].point_z;
        if comm.rank() == 0 {
            println!(
                "Compression: σ₃ = {:.0} Pa, bed height = {:.4} m",
                sigma_3, h
            );
        }

        *state = Some(ConfiningState {
            sigma_3,
            max_velocity: srv.max_velocity,
            gain: srv.gain,
        });
    }

    // Update servo target forces based on current bed height (constant pressure)
    let cs = state.as_ref().unwrap();
    let floor_z = walls.planes[4].point_z;
    let top_z = walls.planes[5].point_z;
    let h = (top_z - floor_z).max(0.001); // avoid division issues
    // Use wall-to-wall spacing, not domain size (domain may be larger for buffer)
    let lx = walls.planes[1].point_x - walls.planes[0].point_x;
    let ly = walls.planes[3].point_y - walls.planes[2].point_y;

    // Walls 0,1 face yz-plane: area = ly × h
    let target_x = cs.sigma_3 * ly * h;
    // Walls 2,3 face xz-plane: area = lx × h
    let target_y = cs.sigma_3 * lx * h;

    for i in 0..2 {
        walls.planes[i].motion = WallMotion::Servo {
            target_force: target_x,
            max_velocity: cs.max_velocity,
            gain: cs.gain,
        };
    }
    for i in 2..4 {
        walls.planes[i].motion = WallMotion::Servo {
            target_force: target_y,
            max_velocity: cs.max_velocity,
            gain: cs.gain,
        };
    }

    // Constant-velocity axial compression
    let top_wall = &mut walls.planes[5];
    top_wall.point_z -= COMPRESS_VEL * dt;
    top_wall.velocity = [0.0, 0.0, -COMPRESS_VEL];
}

/// Accumulator for time-averaging wall stresses between thermo outputs.
#[derive(Default)]
struct StressAccum {
    sigma_1_sum: f64,
    sigma_3_sum: f64,
    count: u64,
}

/// Accumulate stresses every step; write time-averaged values at thermo intervals.
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
    run_state: Res<RunState>,
    thermo: Res<Thermo>,
    input: Res<Input>,
    comm: Res<CommResource>,
    mut file_created: Local<bool>,
    mut initial_height: Local<f64>,
    mut acc: Local<StressAccum>,
) {
    // Compute instantaneous stresses and accumulate (every step)
    // Use wall-to-wall spacing, not domain size (domain may include buffer zones)
    let lx = walls.planes[1].point_x - walls.planes[0].point_x;
    let ly = walls.planes[3].point_y - walls.planes[2].point_y;
    let floor_z = walls.planes[4].point_z;
    let top_z = walls.planes[5].point_z;
    let h = top_z - floor_z;

    let area_top = lx * ly;
    let sigma_1_inst = walls.planes[5].force_accumulator / area_top;

    let f_x_avg =
        (walls.planes[0].force_accumulator + walls.planes[1].force_accumulator) / 2.0;
    let f_y_avg =
        (walls.planes[2].force_accumulator + walls.planes[3].force_accumulator) / 2.0;
    let area_x = ly * h;
    let area_y = lx * h;
    let sigma_3_inst = if h > 0.0 {
        (f_x_avg / area_x + f_y_avg / area_y) / 2.0
    } else {
        0.0
    };

    acc.sigma_1_sum += sigma_1_inst;
    acc.sigma_3_sum += sigma_3_inst;
    acc.count += 1;

    // Only write at thermo intervals (rank 0 only)
    if comm.rank() != 0 {
        return;
    }
    if !run_state.total_cycle.is_multiple_of(thermo.interval) {
        return;
    }

    let output_dir = input.output_dir.as_deref().unwrap_or(".");
    let path = format!("{}/triaxial_stress.csv", output_dir);

    if !*file_created {
        *file_created = true;
        fs::create_dir_all(output_dir).ok();
        let mut f = File::create(&path).expect("cannot create stress CSV");
        writeln!(f, "step,time,strain,sigma_1,sigma_3,q,p")
            .expect("cannot write CSV header");
        *initial_height = top_z - floor_z;
    }

    let step = run_state.total_cycle;
    let time = step as f64 * atoms.dt;

    let strain = if *initial_height > 0.0 {
        (*initial_height - h) / *initial_height
    } else {
        0.0
    };

    // Time-averaged stresses
    let (sigma_1, sigma_3) = if acc.count > 0 {
        (acc.sigma_1_sum / acc.count as f64, acc.sigma_3_sum / acc.count as f64)
    } else {
        (sigma_1_inst, sigma_3_inst)
    };

    // Reset accumulator
    acc.sigma_1_sum = 0.0;
    acc.sigma_3_sum = 0.0;
    acc.count = 0;

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

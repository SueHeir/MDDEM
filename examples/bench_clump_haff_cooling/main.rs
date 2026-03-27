//! Haff's Cooling with Multisphere Clumps
//!
//! Sphere-shaped clumps (7 overlapping sub-spheres approximating a sphere) in a
//! fully periodic box with no gravity. The granular temperature decays following
//! Haff's law: `T(t) = T(0) / (1 + t/tau)^2`.
//!
//! A custom measurement system computes both translational and rotational granular
//! temperatures from body COM velocities and angular velocities, and writes
//! `cooling.csv` with measured values and theoretical Haff predictions.
//!
//! ```bash
//! cargo run --release --example bench_clump_haff_cooling --no-default-features \
//!     -- examples/bench_clump_haff_cooling/config.toml
//! ```

use std::fs;
use std::io::Write;
use std::sync::OnceLock;

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(ClumpPlugin);

    app.add_update_system(measure_cooling, ScheduleSet::PostFinalIntegration);

    app.start();
}

struct HaffParams {
    t0_trans: f64,
    t0_total: f64,
    tau_smooth: f64,
    tau_friction: f64,
    time_origin: f64,
}

static HAFF: OnceLock<HaffParams> = OnceLock::new();

/// Compute translational + rotational granular temperatures and write cooling.csv.
///
/// All temperatures use the mass-weighted velocity-variance convention (units: m²/s²):
///   T_trans = sum(m_i * |v_i - v_mean|²) / (3 * M_total)
///   T_rot   = sum(omega_i . I_i . omega_i) / (3 * M_total)
///   T_total = T_trans + T_rot
fn measure_cooling(
    atoms: Res<Atom>,
    bodies: Res<MultisphereBodyStore>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
) {
    let step = run_state.total_cycle;
    if step % 2000 != 0 {
        return;
    }

    let nbodies = bodies.bodies.len();
    if nbodies == 0 {
        return;
    }

    // ── Translational temperature ──
    let mut m_total = 0.0_f64;
    let mut mv = [0.0_f64; 3];
    for body in &bodies.bodies {
        let m = body.total_mass;
        m_total += m;
        for d in 0..3 {
            mv[d] += m * body.com_vel[d];
        }
    }

    let m_total = comm.all_reduce_sum_f64(m_total);
    let mv = [
        comm.all_reduce_sum_f64(mv[0]),
        comm.all_reduce_sum_f64(mv[1]),
        comm.all_reduce_sum_f64(mv[2]),
    ];

    let v_mean = if m_total > 0.0 {
        [mv[0] / m_total, mv[1] / m_total, mv[2] / m_total]
    } else {
        [0.0; 3]
    };

    let mut ke_trans = 0.0_f64;
    for body in &bodies.bodies {
        let m = body.total_mass;
        let mut dv2 = 0.0;
        for d in 0..3 {
            let dv = body.com_vel[d] - v_mean[d];
            dv2 += dv * dv;
        }
        ke_trans += m * dv2;
    }
    let ke_trans = comm.all_reduce_sum_f64(ke_trans);

    // ── Rotational temperature ──
    let mut ke_rot_2 = 0.0_f64;
    for body in &bodies.bodies {
        let q = body.quaternion;
        let qc = [q[0], -q[1], -q[2], -q[3]];
        let omega_body = dem_clump::quat_rotate(qc, body.omega);
        let qp = body.principal_axes;
        let qpc = [qp[0], -qp[1], -qp[2], -qp[3]];
        let omega_p = dem_clump::quat_rotate(qpc, omega_body);
        ke_rot_2 += body.principal_moments[0] * omega_p[0] * omega_p[0]
            + body.principal_moments[1] * omega_p[1] * omega_p[1]
            + body.principal_moments[2] * omega_p[2] * omega_p[2];
    }
    let ke_rot_2 = comm.all_reduce_sum_f64(ke_rot_2);

    let t_trans = if m_total > 0.0 { ke_trans / (3.0 * m_total) } else { 0.0 };
    let t_rot = if m_total > 0.0 { ke_rot_2 / (3.0 * m_total) } else { 0.0 };
    let t_total = t_trans + t_rot;

    if comm.rank() != 0 {
        return;
    }

    let time = step as f64 * atoms.dt;

    let params = HAFF.get_or_init(|| {
        let d_eff = 2.0 * 1.1e-3;
        let e = 0.9;
        let mu = 0.3;
        let v_rms = (3.0 * t_trans).sqrt().max(1e-10);
        let tau_smooth = d_eff / (v_rms * (1.0 - e * e));
        let gamma = 1.23 * mu;
        let tau_friction = d_eff / (v_rms * (1.0 - e * e + gamma));
        HaffParams {
            t0_trans: t_trans,
            t0_total: t_total,
            tau_smooth,
            tau_friction,
            time_origin: time,
        }
    });

    let dt = time - params.time_origin;
    let haff_smooth = params.t0_trans / (1.0 + dt / params.tau_smooth).powi(2);
    let haff_friction = params.t0_total / (1.0 + dt / params.tau_friction).powi(2);

    let output_dir = match input.output_dir.as_deref() {
        Some(dir) => dir.to_string(),
        None => "examples/bench_clump_haff_cooling/data".to_string(),
    };
    let filepath = format!("{}/cooling.csv", output_dir);

    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        fs::create_dir_all(&output_dir).ok();
        let mut f = fs::File::create(&filepath).expect("Cannot create cooling.csv");
        writeln!(
            f,
            "step,time,T_trans,T_rot,T_total,haff_smooth,haff_friction"
        )
        .unwrap();
    });

    let mut f = fs::OpenOptions::new()
        .append(true)
        .open(&filepath)
        .expect("Cannot open cooling.csv");
    writeln!(
        f,
        "{},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e}",
        step, time, t_trans, t_rot, t_total, haff_smooth, haff_friction
    )
    .unwrap();
}

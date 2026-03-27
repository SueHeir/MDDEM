//! Haff's Cooling — Single Rough Spheres
//!
//! Companion to `bench_clump_haff_cooling`. Same box, material, particle count,
//! and velocity, but with single spheres instead of clumps. Measures both
//! translational and rotational granular temperatures for comparison.
//!
//! ```bash
//! cargo run --release --example bench_sphere_haff_cooling --no-default-features \
//!     -- examples/bench_sphere_haff_cooling/config.toml
//! ```

use std::fs;
use std::io::Write;
use std::sync::OnceLock;

use mddem::prelude::*;
use mddem::dem_atom::DemAtom;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins);

    app.add_update_system(measure_cooling, ScheduleSet::PostFinalIntegration);

    app.start();
}

struct HaffParams {
    t0_trans: f64,
    tau_smooth: f64,
    time_origin: f64,
}

static HAFF: OnceLock<HaffParams> = OnceLock::new();

fn measure_cooling(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
) {
    let step = run_state.total_cycle;
    if step % 2000 != 0 {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 {
        return;
    }

    // ── Translational temperature ──
    let mut m_total = 0.0_f64;
    let mut mv = [0.0_f64; 3];
    for i in 0..nlocal {
        let m = atoms.mass[i];
        m_total += m;
        for d in 0..3 {
            mv[d] += m * atoms.vel[i][d];
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
    for i in 0..nlocal {
        let m = atoms.mass[i];
        let mut dv2 = 0.0;
        for d in 0..3 {
            let dv = atoms.vel[i][d] - v_mean[d];
            dv2 += dv * dv;
        }
        ke_trans += m * dv2;
    }
    let ke_trans = comm.all_reduce_sum_f64(ke_trans);

    // ── Rotational temperature ──
    // T_rot = sum(I * omega²) / (3 * M_total), where I = (2/5)*m*r²
    let dem = registry.get::<DemAtom>();
    let mut ke_rot_2 = 0.0_f64;
    if let Some(ref dem) = dem {
        for i in 0..nlocal {
            let m = atoms.mass[i];
            let r = dem.radius[i];
            let inertia = 0.4 * m * r * r;
            let w = dem.omega[i];
            ke_rot_2 += inertia * (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        }
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
        let d_eff = 2.0 * 0.0011;
        let e = 0.9;
        let v_rms = (3.0 * t_trans).sqrt().max(1e-10);
        let tau_smooth = d_eff / (v_rms * (1.0 - e * e));
        HaffParams {
            t0_trans: t_trans,
            tau_smooth,
            time_origin: time,
        }
    });

    let dt = time - params.time_origin;
    let haff_smooth = params.t0_trans / (1.0 + dt / params.tau_smooth).powi(2);

    let output_dir = match input.output_dir.as_deref() {
        Some(dir) => dir.to_string(),
        None => "examples/bench_sphere_haff_cooling/data".to_string(),
    };
    let filepath = format!("{}/cooling.csv", output_dir);

    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        fs::create_dir_all(&output_dir).ok();
        let mut f = fs::File::create(&filepath).expect("Cannot create cooling.csv");
        writeln!(f, "step,time,T_trans,T_rot,T_total,haff_smooth").unwrap();
    });

    let mut f = fs::OpenOptions::new()
        .append(true)
        .open(&filepath)
        .expect("Cannot open cooling.csv");
    writeln!(
        f,
        "{},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e}",
        step, time, t_trans, t_rot, t_total, haff_smooth
    )
    .unwrap();
}

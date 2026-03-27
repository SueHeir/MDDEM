//! Multisphere segregation benchmark — validates that dimer clumps rise above
//! single spheres under vertical vibration (Brazil nut effect for shape).
//!
//! A 50/50 mixture of single spheres (type 0) and dimer clumps (type 1) is
//! placed in a box with a vertically oscillating floor, periodic lateral walls,
//! and a static ceiling.  Under vibration at Γ = A·ω²/g ≈ 3.6, the dimers'
//! larger effective size causes them to segregate upward.
//!
//! A custom system writes `segregation.csv` every N steps with the center-of-mass
//! heights of spheres vs dimers and the segregation index S.
//!
//! ```bash
//! cargo run --release --example bench_multisphere_segregation --no-default-features \
//!     -- examples/bench_multisphere_segregation/config.toml
//! ```

use std::fs;
use std::io::Write;

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(GravityPlugin)
        .add_plugins(WallPlugin)
        .add_plugins(ClumpPlugin);

    // Measure segregation periodically
    app.add_update_system(measure_segregation, ScheduleSet::PostFinalIntegration);

    app.start();
}

/// Measure and write segregation data: average z of spheres vs dimers.
///
/// Single spheres have type 0. Dimer bodies are identified via MultisphereBodyStore.
/// Sub-sphere atoms (body_id > 0) are excluded from single-sphere averages.
fn measure_segregation(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    bodies: Res<MultisphereBodyStore>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
) {
    let step = run_state.total_cycle;

    // Write every 500 steps
    if step % 500 != 0 {
        return;
    }

    let clump = registry.get::<ClumpAtom>();

    let nlocal = atoms.nlocal as usize;

    // Accumulate z-positions weighted by mass for each species
    let mut sphere_z_sum = 0.0_f64;
    let mut sphere_mass_sum = 0.0_f64;
    let mut dimer_z_sum = 0.0_f64;
    let mut dimer_mass_sum = 0.0_f64;

    // Single spheres: type 0, not part of any body
    for i in 0..nlocal {
        if atoms.atom_type[i] != 0 {
            continue;
        }
        // Skip body sub-spheres
        if let Some(ref c) = clump {
            if i < c.body_id.len() && c.body_id[i] > 0.0 {
                continue;
            }
        }
        sphere_z_sum += atoms.mass[i] * atoms.pos[i][2];
        sphere_mass_sum += atoms.mass[i];
    }

    // Dimer bodies: use COM z from MultisphereBodyStore
    for body in &bodies.bodies {
        dimer_z_sum += body.total_mass * body.com_pos[2];
        dimer_mass_sum += body.total_mass;
    }

    // Global reduction
    let sphere_z_total = comm.all_reduce_sum_f64(sphere_z_sum);
    let sphere_mass_total = comm.all_reduce_sum_f64(sphere_mass_sum);
    let dimer_z_total = comm.all_reduce_sum_f64(dimer_z_sum);
    let dimer_mass_total = comm.all_reduce_sum_f64(dimer_mass_sum);

    if comm.rank() != 0 {
        return;
    }

    let z_sphere = if sphere_mass_total > 0.0 {
        sphere_z_total / sphere_mass_total
    } else {
        0.0
    };
    let z_dimer = if dimer_mass_total > 0.0 {
        dimer_z_total / dimer_mass_total
    } else {
        0.0
    };

    // Segregation index: S = (z_dimer - z_sphere) / (z_dimer + z_sphere)
    let seg_index = if (z_dimer + z_sphere) > 0.0 {
        (z_dimer - z_sphere) / (z_dimer + z_sphere)
    } else {
        0.0
    };

    let time = step as f64 * atoms.dt;

    let output_dir = match input.output_dir.as_deref() {
        Some(dir) => dir.to_string(),
        None => "data".to_string(),
    };
    let filepath = format!("{}/segregation.csv", output_dir);

    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        fs::create_dir_all(&output_dir).ok();
        let mut f = fs::File::create(&filepath).expect("Cannot create segregation.csv");
        writeln!(f, "step,time,z_sphere,z_dimer,segregation_index").unwrap();
    });

    let mut f = fs::OpenOptions::new()
        .append(true)
        .open(&filepath)
        .expect("Cannot open segregation.csv");
    writeln!(
        f,
        "{},{:.8e},{:.8e},{:.8e},{:.8e}",
        step, time, z_sphere, z_dimer, seg_index
    )
    .unwrap();
}

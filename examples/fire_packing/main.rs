//! Two-stage example: FIRE packing → Haff's cooling.
//!
//! Stage 1 ("minimize"): Randomly inserts ~120 overlapping spheres at ~50% volume
//! fraction, then uses FIRE energy minimization to push them apart until all
//! forces are below `ftol`.  `save_at_end = true` writes dump + restart files
//! when FIRE converges.
//!
//! Stage 2 ("cooling"): Assigns random velocities and runs standard Velocity
//! Verlet integration. With inelastic contacts (e = 0.5) and no energy input,
//! kinetic energy decays following Haff's law: KE ~ t^{-2}.
//!
//! # Two-run workflow
//!
//! ```sh
//! # Run 1: full FIRE + cooling (saves restart at end of minimize)
//! cargo run --example fire_packing
//!
//! # Run 2: skip FIRE, load restart, run cooling only
//! cargo run --example fire_packing -- examples/fire_packing/config_skip.toml
//! ```

use mddem::prelude::*;
use mddem::dem_granular::{
    GranularTempPlugin, HertzMindlinContactPlugin, RotationalDynamicsPlugin,
};
use mddem_fire::FireMinPlugin;

fn main() {
    let mut app = App::new();

    app.add_plugins(CorePlugins);

    // Granular physics — individual plugins with stage-gated integrators.
    app.add_plugins(DemAtomPlugin)
        .add_plugins(DemAtomInsertPlugin)
        .add_plugins(VelocityVerletPlugin::for_stage("cooling"))
        .add_plugins(HertzMindlinContactPlugin)
        .add_plugins(RotationalDynamicsPlugin)
        .add_plugins(GranularTempPlugin);

    // FIRE in "minimize" stage only
    app.add_plugins(FireMinPlugin::for_stage("minimize"));

    // Assign random velocities once at the start of the cooling stage
    app.add_update_system(
        assign_random_velocities.run_if(in_stage("cooling")),
        ScheduleSet::PreInitialIntegration,
    );

    app.start();
}

/// One-shot system: assigns random velocities to all particles at the start of
/// the cooling stage, then disables itself via a local flag.
fn assign_random_velocities(mut atoms: ResMut<Atom>, mut done: Local<bool>) {
    if *done {
        return;
    }
    *done = true;

    let nlocal = atoms.nlocal as usize;
    let speed = 0.1; // m/s
    for i in 0..nlocal {
        let tag = atoms.tag[i] as f64;
        let theta = 2.39996 * tag; // golden angle
        let phi = (tag * 0.618033).fract() * std::f64::consts::PI;
        atoms.vel[i][0] = speed * phi.sin() * theta.cos();
        atoms.vel[i][1] = speed * phi.sin() * theta.sin();
        atoms.vel[i][2] = speed * phi.cos();
    }
}

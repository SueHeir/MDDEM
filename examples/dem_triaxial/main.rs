//! Triaxial compression: insert → relax → compress z-axis with fix deform.
//!
//! Demonstrates box deformation (engineering strain rate) as an alternative
//! to servo walls for granular compression tests.
//!
//! ```bash
//! cargo run --example dem_triaxial --no-default-features -- examples/dem_triaxial/config.toml
//! ```

use mddem::prelude::*;

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
        .add_plugins(DeformPlugin)
        .add_plugins(StatesPlugin::new(Phase::Insert, ScheduleSet::PostFinalIntegration))
        .add_plugins(StageAdvancePlugin::<Phase>::new(ScheduleSet::PostFinalIntegration));

    app.add_update_system(
        check_insert_settled.run_if(in_state(Phase::Insert)),
        ScheduleSet::PostFinalIntegration,
    );
    app.add_update_system(
        check_relaxed.run_if(in_state(Phase::Relax)),
        ScheduleSet::PostFinalIntegration,
    );
    app.add_update_system(
        report_strain.run_if(in_state(Phase::Compress)),
        ScheduleSet::PostFinalIntegration,
    );

    app.start();
}

/// During insert stage, wait for KE to drop below threshold then advance to relax.
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

/// During relax stage, wait for KE to drop further then advance to compress.
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

/// Report axial strain and box dimensions during compression.
fn report_strain(
    domain: Res<Domain>,
    deform_state: Res<DeformState>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
) {
    if comm.rank() != 0 {
        return;
    }
    let step = run_state.total_cycle;
    if step % 5000 != 0 {
        return;
    }

    let z_lo = domain.boundaries_low[2];
    let z_hi = domain.boundaries_high[2];
    let z_size = z_hi - z_lo;

    // Compute engineering strain if z-axis is deforming
    if let Some(ref ax) = deform_state.axes[2] {
        let l0 = ax.hi_0 - ax.lo_0;
        let strain = (z_size - l0) / l0;
        println!(
            "Step {}: z=[{:.6}, {:.6}], Lz={:.6}, strain={:.6e}, vol={:.6e}",
            step, z_lo, z_hi, z_size, strain, domain.volume
        );
    }
}

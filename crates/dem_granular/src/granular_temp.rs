//! Granular temperature output — measures velocity fluctuations in the system.
//!
//! Granular temperature `T_g` quantifies the kinetic energy of velocity fluctuations
//! (deviations from the mean flow velocity):
//!
//! ```text
//! T_g = Σ m_i |v_i - v_mean|² / (3 M_total)
//! ```
//!
//! This module writes `T_g`, total kinetic energy, and total momentum magnitude
//! to `data/GranularTemp.txt` at each thermo output interval. The file is
//! truncated at step 0 and appended thereafter.
//!
//! # Output format
//!
//! Each line: `step  time  T_granular  KE_total  |p_total|`

use std::{
    fs::{self, OpenOptions},
    io::Write,
};

use grass_app::prelude::*;
use grass_scheduler::prelude::*;

use mddem_core::{Atom, CommResource, Input, RunConfig, RunState, ParticleSimScheduleSet};

/// Plugin that outputs granular temperature to `data/GranularTemp.txt`.
///
/// Registers [`print_granular_temperature`] at [`ParticleSimScheduleSet::PreExchange`],
/// writing one line per thermo interval with step, time, granular temperature,
/// total kinetic energy, and total momentum magnitude.
pub struct GranularTempPlugin;

impl Plugin for GranularTempPlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(print_granular_temperature, ParticleSimScheduleSet::PreExchange);
    }
}

/// Compute and write granular temperature, KE, and momentum to file.
///
/// All ranks participate in the allreduce; only rank 0 writes the file.
pub fn print_granular_temperature(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
    input: Res<Input>,
) {
    let index = scheduler_manager.index;
    let thermo_interval = run_config.current_stage(index).thermo;
    if !run_state.total_cycle.is_multiple_of(thermo_interval) {
        return;
    }
    let nlocal = atoms.nlocal as usize;
    let mut local_mv_x = 0.0;
    let mut local_mv_y = 0.0;
    let mut local_mv_z = 0.0;
    let mut local_mass = 0.0;
    for i in 0..nlocal {
        local_mv_x += atoms.mass[i] * atoms.vel[i][0];
        local_mv_y += atoms.mass[i] * atoms.vel[i][1];
        local_mv_z += atoms.mass[i] * atoms.vel[i][2];
        local_mass += atoms.mass[i];
    }
    let global_mv_x = comm.all_reduce_sum_f64(local_mv_x);
    let global_mv_y = comm.all_reduce_sum_f64(local_mv_y);
    let global_mv_z = comm.all_reduce_sum_f64(local_mv_z);
    let global_mass = comm.all_reduce_sum_f64(local_mass);
    let avg_vx = global_mv_x / global_mass;
    let avg_vy = global_mv_y / global_mass;
    let avg_vz = global_mv_z / global_mass;
    let mut vel_diff = 0.0;
    for i in 0..nlocal {
        vel_diff += atoms.mass[i] * (atoms.vel[i][0] - avg_vx).powi(2)
            + atoms.mass[i] * (atoms.vel[i][1] - avg_vy).powi(2)
            + atoms.mass[i] * (atoms.vel[i][2] - avg_vz).powi(2);
    }
    let vel_diff_sum = comm.all_reduce_sum_f64(vel_diff);
    let granular_temperature = vel_diff_sum / (3.0 * global_mass);

    // KE computation — all ranks must participate in allreduce
    let mut local_ke = 0.0;
    for i in 0..nlocal {
        local_ke += 0.5
            * atoms.mass[i]
            * (atoms.vel[i][0].powi(2) + atoms.vel[i][1].powi(2) + atoms.vel[i][2].powi(2));
    }
    let global_ke = comm.all_reduce_sum_f64(local_ke);

    // Total momentum magnitude (should be ~conserved)
    let mom_mag = (global_mv_x * global_mv_x + global_mv_y * global_mv_y + global_mv_z * global_mv_z).sqrt();

    if comm.rank() != 0 {
        return;
    }
    let physical_time = run_state.total_cycle as f64 * atoms.dt;
    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("{}/data", dir),
        None => "data".to_string(),
    };
    let result = fs::create_dir_all(&base_dir);
    if let Err(_error) = result {
        println!("Could not create file directory {}", base_dir)
    }
    let data_path = format!("{}/GranularTemp.txt", base_dir);
    let mut file = if run_state.total_cycle == 0 {
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&data_path)
            .expect("failed to create GranularTemp.txt — check output directory permissions")
    } else {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&data_path)
            .expect("failed to open GranularTemp.txt for append — check file permissions")
    };

    writeln!(
        &mut file,
        "{} {:.6e} {:.10e} {:.10e} {:.10e}",
        run_state.total_cycle, physical_time, granular_temperature, global_ke, mom_mag
    )
    .expect("failed to write to GranularTemp.txt — disk may be full");
}

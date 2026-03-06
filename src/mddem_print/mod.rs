
use std::{fs::{self, File, OpenOptions}, io::Write, time::Instant};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use mpi::collective::SystemOperation;
use mpi::traits::CommunicatorCollectives;
use nalgebra::Vector3;

use crate::{
    mddem_atom::Atom,
    mddem_communication::Comm,
    mddem_input::Input,
    mddem_neighbor::Neighbor,
    mddem_verlet::Verlet,
};

pub struct PrintPlugin;

impl Plugin for PrintPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Thermo::new())
            .add_setup_system(setup_thermo, ScheduleSetupSet::PostSetup)
            .add_update_system(print_vtp, ScheduleSet::PostFinalIntegration)
            .add_update_system(print_thermo, ScheduleSet::PostFinalIntegration)
            .add_update_system(print_granular_temperature, ScheduleSet::PreExchange);
    }
}


// ── Thermo ──────────────────────────────────────────────────────────────────

pub struct Thermo {
    pub interval: usize,
    start_time: Instant,
    last_printed_step: usize,
}

impl Thermo {
    pub fn new() -> Self {
        Thermo {
            interval: 100,
            start_time: Instant::now(),
            last_printed_step: 0,
        }
    }
}

/// Reads `thermo N` from input, resets the wall-clock timer, and prints the
/// column header so output is aligned with LAMMPS-style thermo logging.
pub fn setup_thermo(
    input: Res<Input>,
    scheduler_manager: Res<SchedulerManager>,
    comm: Res<Comm>,
    verlet: Res<Verlet>,
    mut thermo: ResMut<Thermo>,
) {
    let commands = &input.current_commands[scheduler_manager.index];
    for c in commands.iter() {
        let values = c.split_whitespace().collect::<Vec<&str>>();
        if values.len() >= 2 && values[0] == "thermo" {
            thermo.interval = values[1].parse::<usize>().unwrap();
        }
    }

    thermo.start_time = Instant::now();
    thermo.last_printed_step = verlet.total_cycle;

    if comm.rank == 0 {
        println!();
        println!(
            "{:<12} {:<8} {:<14} {:<12} {:<14} {:<12}",
            "Step", "Atoms", "KE(J)", "Neighbors", "WallTime(s)", "Step/s"
        );
        println!("{}", "-".repeat(74));
    }
}

/// Prints one thermo line every `thermo.interval` steps.
/// Quantities: total step, global atom count, global translational KE (real
/// atoms only), total neighbor pairs, elapsed wall time since last print, and
/// steps per second.
pub fn print_thermo(
    atoms: Res<Atom>,
    verlet: Res<Verlet>,
    comm: Res<Comm>,
    neighbor: Res<Neighbor>,
    mut thermo: ResMut<Thermo>,
) {
    let step = verlet.total_cycle;
    if step % thermo.interval != 0 {
        return;
    }

    // KE over real (non-ghost) atoms only
    let local_ke: f64 = atoms.velocity[..atoms.nlocal as usize]
        .iter()
        .zip(atoms.mass[..atoms.nlocal as usize].iter())
        .map(|(v, m)| 0.5 * m * v.norm_squared())
        .sum();

    let local_neighbors = neighbor.neighbor_list.len() as f64;

    let mut global_ke = 0.0f64;
    comm.world.all_reduce_into(&local_ke, &mut global_ke, SystemOperation::sum());

    let mut global_neighbors_f = 0.0f64;
    comm.world.all_reduce_into(&local_neighbors, &mut global_neighbors_f, SystemOperation::sum());
    let global_neighbors = global_neighbors_f as usize;

    if comm.rank == 0 {
        let elapsed = thermo.start_time.elapsed().as_secs_f64();
        let steps_since = (step - thermo.last_printed_step) as f64;
        let steps_per_sec = if elapsed > 1e-9 { steps_since / elapsed } else { 0.0 };

        println!(
            "{:<12} {:<8} {:<14.6e} {:<12} {:<14.4} {:<12.1}",
            step, atoms.natoms, global_ke, global_neighbors, elapsed, steps_per_sec
        );

        thermo.start_time = Instant::now();
        thermo.last_printed_step = step;
    }
}


// ── VTP output ───────────────────────────────────────────────────────────────

pub fn print_vtp(atoms: Res<Atom>, verlet: Res<Verlet>, comm: Res<Comm>) {
    let count = verlet.total_cycle;
    let rank = comm.rank;

    if count % 2000 != 0 {
        return
    }
    let filename = format!("./vtp/{}CYCLE_{}RANK.vtp", count, rank);

    let result = fs::create_dir_all("./vtp");
    if let Err(_error) = result {
        println!("Could not create file directory ./vtp")
    }
    let mut file = File::create(filename).unwrap();

    write!(&mut file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n<PolyData>\n").unwrap();

    write!(
        &mut file,
        "<Piece NumberOfPoints=\"{}\">\n",
        atoms.pos.len()
    )
    .unwrap();

    write!(&mut file, "<Points>").unwrap();

    write!(
        &mut file,
        "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">"
    )
    .unwrap();
    for i in 0..atoms.pos.len() {
        writeln!(
            &mut file,
            "{} {} {}",
            atoms.pos[i][0], atoms.pos[i][1], atoms.pos[i][2]
        )
        .unwrap();
    }
    write!(&mut file, "</DataArray>\n").unwrap();
    write!(&mut file, "</Points>\n").unwrap();

    write!(&mut file, "<PointData Scalars=\"\" Vectors=\"\">\n").unwrap();
    write!(
        &mut file,
        "<DataArray type=\"Float32\" Name=\"Radius\" format=\"ascii\">\n"
    )
    .unwrap();
    for i in 0..atoms.pos.len() {
        writeln!(&mut file, "{}", atoms.skin[i]).unwrap();
    }

    write!(&mut file, "</DataArray>").unwrap();

    write!(
        &mut file,
        "<DataArray type=\"Float32\" Name=\"Vel_Mag\" format=\"ascii\">\n"
    )
    .unwrap();
    for i in 0..atoms.pos.len() {
        writeln!(&mut file, "{}", atoms.velocity[i].norm()).unwrap();
    }

    write!(&mut file, "</DataArray>").unwrap();

    write!(
        &mut file,
        "<DataArray type=\"Int32\" Name=\"IsGhost\" format=\"ascii\">\n"
    )
    .unwrap();
    for i in 0..atoms.pos.len() {
        let mut is_ghost = 0;
        if i >= atoms.nlocal.try_into().unwrap() {
            is_ghost = 1
        }
        writeln!(&mut file, "{}", is_ghost).unwrap();
    }

    write!(&mut file, "</DataArray>").unwrap();

    write!(
        &mut file,
        "<DataArray type=\"Int32\" Name=\"IsCollision\" format=\"ascii\">\n"
    )
    .unwrap();
    for i in 0..atoms.pos.len() {
        let mut is_collision = 0;
        if atoms.is_collision[i] {
            is_collision = 1
        }
        writeln!(&mut file, "{}", is_collision).unwrap();
    }

    write!(&mut file, "</DataArray>").unwrap();

    write!(
        &mut file,
        "</PointData>\n</Piece>\n</PolyData>\n</VTKFile>\n"
    )
    .unwrap();
}


// ── Granular temperature (written to file) ───────────────────────────────────
//
// Option A: snapshot T at each output step.  No MPI runs on non-output steps.

pub fn print_granular_temperature(
    atoms: Res<Atom>,
    verlet: Res<Verlet>,
    comm: Res<Comm>,
    thermo: Res<Thermo>,
) {
    // Skip non-output steps entirely — zero MPI, zero arithmetic.
    if verlet.total_cycle % thermo.interval != 0 {
        return;
    }

    // ── Pass 1: global centre-of-mass velocity ───────────────────────────────
    let mut local_mass_velocity = Vector3::zeros();
    let mut local_mass = 0.0;

    for i in 0..atoms.nlocal as usize {
        local_mass_velocity += atoms.mass[i] * atoms.velocity[i];
        local_mass += atoms.mass[i];
    }

    let mut global_mv_x = 0.0;
    let mut global_mv_y = 0.0;
    let mut global_mv_z = 0.0;
    let mut global_mass = 0.0;

    comm.world.all_reduce_into(&local_mass_velocity.x, &mut global_mv_x, SystemOperation::sum());
    comm.world.all_reduce_into(&local_mass_velocity.y, &mut global_mv_y, SystemOperation::sum());
    comm.world.all_reduce_into(&local_mass_velocity.z, &mut global_mv_z, SystemOperation::sum());
    comm.world.all_reduce_into(&local_mass, &mut global_mass, SystemOperation::sum());

    let global_mass_velocity = Vector3::new(global_mv_x, global_mv_y, global_mv_z);
    let mass_ave_velocity = global_mass_velocity / global_mass;

    // ── Pass 2: vel_diff relative to global <v>, then reduce ────────────────
    let mut vel_diff = 0.0;
    for i in 0..atoms.nlocal as usize {
        let vel = atoms.velocity[i];
        vel_diff += atoms.mass[i] * (vel.x - mass_ave_velocity.x).powi(2)
            + atoms.mass[i] * (vel.y - mass_ave_velocity.y).powi(2)
            + atoms.mass[i] * (vel.z - mass_ave_velocity.z).powi(2);
    }

    let mut vel_diff_sum = 0.0;
    comm.world.all_reduce_into(&vel_diff, &mut vel_diff_sum, SystemOperation::sum());

    // T = Σ mᵢ(vᵢ−⟨v⟩)² / (3 · Σmᵢ)
    let granular_temperature = vel_diff_sum / (3.0 * global_mass);

    if comm.rank != 0 {
        return;
    }

    let physical_time = verlet.total_cycle as f64 * atoms.dt;

    let result = fs::create_dir_all("./data");
    if let Err(_error) = result {
        println!("Could not create file directory ./data")
    }
    let mut file = if verlet.total_cycle == 0 {
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("./data/GranularTemp.txt")
            .unwrap()
    } else {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open("./data/GranularTemp.txt")
            .unwrap()
    };

    write!(&mut file, "{} {:.6e} {:.10e}\n", verlet.total_cycle, physical_time, granular_temperature).unwrap();
}

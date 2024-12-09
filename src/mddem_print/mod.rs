
use std::{fs::File, io::Write};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use crate::{mddem_atom::Atom, mddem_communication::Comm, mddem_verlet::Verlet};

pub struct PrintPlugin;

impl Plugin for PrintPlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(print_vtp, ScheduleSet::PostFinalIntegration)
            .add_update_system(print_cycle_count, ScheduleSet::PostFinalIntegration);
    }
}



pub fn print_vtp(atoms: Res<Atom>, verlet: Res<Verlet>, comm: Res<Comm>) {
    let count = verlet.total_cycle;
    let rank = comm.rank;

    if count % 2000 != 0 {
        return
    }
    let filename = format!("./vtp/{}CYCLE_{}RANK.vtp", count, rank);
    let mut file = File::create(filename).unwrap();

    // Write a &str in the file (ignoring the result).
    write!(&mut file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n<PolyData>\n").unwrap();

    write!(
        &mut file,
        "<Piece NumberOfPoints=\"{}\">\n",
        atoms.radius.len()
    )
    .unwrap();

    write!(&mut file, "<Points>").unwrap();

    write!(
        &mut file,
        "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">"
    )
    .unwrap();
    for i in 0..atoms.radius.len() {
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
    for i in 0..atoms.radius.len() {
        writeln!(&mut file, "{}", atoms.radius[i]).unwrap();
    }

    write!(&mut file, "</DataArray>").unwrap();


    write!(
        &mut file,
        "<DataArray type=\"Float32\" Name=\"Vel_Mag\" format=\"ascii\">\n"
    )
    .unwrap();
    for i in 0..atoms.radius.len() {
        writeln!(&mut file, "{}", atoms.velocity[i].norm()).unwrap();
    }

    write!(&mut file, "</DataArray>").unwrap();


    write!(
        &mut file,
        "<DataArray type=\"Int32\" Name=\"IsGhost\" format=\"ascii\">\n"
    )
    .unwrap();
    for i in 0..atoms.radius.len() {
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
    for i in 0..atoms.radius.len() {
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



pub fn print_cycle_count(verlet: Res<Verlet>, comm: Res<Comm>) { 
    if verlet.total_cycle % 10000 != 0 {
        return
    }
    if comm.rank == 0 {
        println!("Cycle: {}", verlet.total_cycle)
    }
    
}

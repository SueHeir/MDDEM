use std::{collections::HashMap, hash::Hash};

use nalgebra::Vector3;

use super::{
    atom::{Atom, AtomMPI}, comm::Comm, domain::Domain, input::Input, scheduler::{Res, ResMut, ScheduleSet::*, Scheduler}
};



pub struct NeighborData {
    pub distance: f64,
}

pub struct Neighbor {
    pub skin_fraction: f64,
    pub neighbor_list_map: HashMap<(usize, usize), NeighborData>
}


impl Neighbor {
    pub fn new() -> Self {
        Neighbor {
            skin_fraction: 1.0,
            neighbor_list_map: HashMap::new()
        }
    }
}


pub fn neighbor_app(scheduler: &mut Scheduler) {
    scheduler.add_resource(Neighbor::new());
    scheduler.add_setup_system(read_input, Setup);
    scheduler.add_update_system(brute_force_neighbor_list, Neighbor);


}

pub fn read_input(input: Res<Input>, mut neighbor: ResMut<Neighbor>, comm: Res<Comm>,) {
    let commands = &input.commands;
    for c in commands.iter() {
        let values = c.split_whitespace().collect::<Vec<&str>>();

        if values.len() > 0 {
            match values[0] {
                "neighbor" => {
                    if comm.rank == 0 {
                        println!("Comm: {}", c);
                    }

                    neighbor.skin_fraction = values[1].parse::<f64>().unwrap();
                }
                _ => {}
            }
        }
    }

}

// pub fn setup(mut neighbor: ResMut<Neighbor>) {

// }




pub fn brute_force_neighbor_list(atoms: Res<Atom>, mut neighbor: ResMut<Neighbor>) {
    neighbor.neighbor_list_map.clear();

    for i in 0..atoms.radius.len() {
        for j in (i+1)..atoms.radius.len() {
            if atoms.tag[i] == atoms.tag[j] {
                continue;
            }

            let p1 = atoms.pos[i];
            let p2 = atoms.pos[j];
            let r1 = atoms.radius[i];
            let r2 = atoms.radius[j];

            let position_difference = p2 - p1;
            let distance = position_difference.norm();

            if distance < (r1 + r2)*neighbor.skin_fraction {

                neighbor.neighbor_list_map.insert((i,j), NeighborData { distance });
            }
        }
    }
}

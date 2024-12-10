use std::collections::HashMap;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use crate::{mddem_atom::Atom, mddem_communication::Comm, mddem_input::Input};


pub struct NeighborPlugin;

impl Plugin for NeighborPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Neighbor::new())
            .add_setup_system(read_input, ScheduleSetupSet::Setup)
            .add_update_system(brute_force_neighbor_list, ScheduleSet::Neighbor);
    }
}



pub struct NeighborData {
    pub _distance: f64,
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



pub fn read_input(input: Res<Input>, scheduler_manager: Res<SchedulerManager>, mut neighbor: ResMut<Neighbor>, comm: Res<Comm>,) {
    let commands = &input.current_commands[scheduler_manager.index];
    for c in commands.iter() {
        let values = c.split_whitespace().collect::<Vec<&str>>();

        if values.len() > 0 {
            match values[0] {
                "neighbor" => {
                    if comm.rank == 0 {
                        println!("Neighbor: {}", c);
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

    for i in 0..atoms.pos.len() {
        for j in (i+1)..atoms.pos.len() {
            if atoms.tag[i] == atoms.tag[j] {
                continue;
            }

            let p1 = atoms.pos[i];
            let p2 = atoms.pos[j];
            let r1 = atoms.skin[i];
            let r2 = atoms.skin[j];

            let position_difference = p2 - p1;
            let distance = position_difference.norm();

            if distance < (r1 + r2)*neighbor.skin_fraction {

                neighbor.neighbor_list_map.insert((i,j), NeighborData { _distance: distance });
            }
        }
    }
}

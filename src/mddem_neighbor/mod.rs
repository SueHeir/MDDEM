use std::collections::HashMap;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;

use crate::{mddem_atom::Atom, mddem_communication::Comm, mddem_domain::Domain, mddem_input::Input};


pub struct NeighborPlugin {
    pub brute_force: bool,
}

impl Plugin for NeighborPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Neighbor::new())
            .add_setup_system(read_input, ScheduleSetupSet::Setup)
            .add_setup_system(setup, ScheduleSetupSet::PostSetup); // Needs to be called after Domain


        if self.brute_force {
            app.add_update_system(brute_force_neighbor_list, ScheduleSet::Neighbor);
        } else {
            app.add_update_system(sweep_and_prune_neighbor_list, ScheduleSet::Neighbor);
        }
           
    }
}



pub struct Bin {
    pub inside_atom_indexs: Vec<usize>,
    pub outside_atom_indexs: Vec<usize>,
}

impl Bin {
    pub fn new() -> Self {
        Bin { 
            inside_atom_indexs: Vec::new(),
            outside_atom_indexs: Vec::new()
         }
    }
}

pub struct Neighbor {
    pub skin_fraction: f64,
    pub bins: HashMap<(i32,i32, i32), Bin>,
    pub neighbor_list: Vec<(usize, usize)>,
    pub sweep_and_prune: Vec<(usize, f64)>,
    pub bin_min_size: f64,
    pub bin_size: Vector3<f64>,
    pub bin_count: Vector3<i32>,
    /// Positions of all atoms (local + ghost) at the time of the last neighbor build.
    pub last_build_pos: Vec<Vector3<f64>>,
    /// Tags of all atoms at the time of the last neighbor build.
    /// Used to detect ghost identity changes between builds.
    pub last_build_tags: Vec<u32>,
    pub steps_since_build: usize,
}


impl Neighbor {
    pub fn new() -> Self {
        Neighbor {
            skin_fraction: 1.0,
            bins: HashMap::new(),
            neighbor_list: Vec::new(),
            sweep_and_prune: Vec::new(),
            bin_min_size: 1.0,
            bin_size: Vector3::new(1.0,1.0,1.0),
            bin_count: Vector3::new(1,1,1),
            last_build_pos: Vec::new(),
            last_build_tags: Vec::new(),
            steps_since_build: 0,
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
                    neighbor.bin_min_size = values[2].parse::<f64>().unwrap();
                }
                _ => {}
            }
        }
    }
}

pub fn setup(mut neighbor: ResMut<Neighbor>, domain: Res<Domain>) {
    let whole_number_of_bins = domain.sub_length / neighbor.bin_min_size;

    let xi = whole_number_of_bins.x.floor() as i32;
    let yi = whole_number_of_bins.y.floor() as i32;
    let zi = whole_number_of_bins.z.floor() as i32;

    neighbor.bin_count = Vector3::new(xi + 2,yi + 2,zi + 2); // + 2 for ghost atoms outside subdomain, one for each size
    neighbor.bin_size = Vector3::new(domain.sub_length.x / xi as f64, domain.sub_length.y / yi as f64,domain.sub_length.z / zi as f64);


    println!("{:?} {:?}", neighbor.bin_count, neighbor.bin_size);
    for x in 0..xi+2 {
        for y in 0..yi+2 {
            for z in 0..zi+2 {
                neighbor.bins.insert((x,y,z), Bin::new());
            }
        }
    }

//     for ((x,y,z), bin) in &neighbor.bins {
//         println!("{} {} {}", x,y,z);
//     }
}

pub fn bin_based_neighbor_list(atoms: Res<Atom>, mut neighbor: ResMut<Neighbor>, domain: Res<Domain>) {
    //Clear Neighbor data
    neighbor.neighbor_list.clear();
    for ((_,_,_), bin) in &mut neighbor.bins {
        bin.inside_atom_indexs.clear();
        bin.outside_atom_indexs.clear();
    }


    // Organize Atoms into bins
    for i in 0..atoms.pos.len() {
        // A ghost atom outside the subdomain at the beginning will have an interger vaule of 0
        let pos = atoms.pos[i] - domain.sub_domain_low + neighbor.bin_size;

        let xi = (pos.x / neighbor.bin_size.x).floor() as i32;
        let yi = (pos.y / neighbor.bin_size.y).floor() as i32;
        let zi = (pos.z / neighbor.bin_size.z).floor() as i32;


       
            
        for x in -1..2 {
            for y in -1..2 {
                for z in -1..2 {
                    if x == 0 && y == 0 && z == 0 {
                        if let Some(bin) = neighbor.bins.get_mut(&(xi,yi,zi)) {
                            bin.inside_atom_indexs.push(i);
                        } else {
                            println!("nonbin {} {} {}", xi, yi,zi);
                        }
                        
                    } else if let Some(bin) = neighbor.bins.get_mut(&(xi + x, yi + y, zi + z)) {
                        bin.outside_atom_indexs.push(i);
                    } // else isn't needed hear because the most left bin doesn't have a bin to the left of it (The cases where the option is None)
                }
            }
        }

    }

    //Loop over bins which contain only real atoms (not first or last bin in each direction)
    for x in 1..(neighbor.bin_count.x-2) {
        for y in 1..(neighbor.bin_count.y-2) {
            for z in 1..(neighbor.bin_count.z-2) {
                



                let inside = neighbor.bins.get(&(x,y,z)).unwrap().inside_atom_indexs.clone(); //Need way around borrow checker here?
                let outside = neighbor.bins.get(&(x,y,z)).unwrap().outside_atom_indexs.clone();

                //Check inside atoms vs inside atoms
                for i in 0..inside.len() {
                    for j in (i+1)..inside.len() {
                        let ai = inside[i];
                        let aj = inside[j];
                        if atoms.tag[ai] == atoms.tag[aj] {
                            continue;
                        }

                        let p1 = atoms.pos[ai];
                        let p2 = atoms.pos[aj];
                        let r1 = atoms.skin[ai];
                        let r2 = atoms.skin[aj];

                        let position_difference = p2 - p1;
                        let distance = position_difference.norm();

                        if distance < (r1 + r2)*neighbor.skin_fraction {
                            neighbor.neighbor_list.push((ai, aj));
                        }
                    }
                }
                //Check inside atoms vs outside atoms
                for i in 0..inside.len() {
                    for j in 0..outside.len() {
                        let ai = inside[i];
                        let aj = outside[j];
                        if atoms.tag[ai] == atoms.tag[aj] || (atoms.is_ghost[ai] && atoms.is_ghost[aj]) {
                            continue;
                        }

                        let p1 = atoms.pos[ai];
                        let p2 = atoms.pos[aj];
                        let r1 = atoms.skin[ai];
                        let r2 = atoms.skin[aj];

                        let position_difference = p2 - p1;
                        let distance = position_difference.norm();

                        if distance < (r1 + r2)*neighbor.skin_fraction {
                            neighbor.neighbor_list.push((ai, aj));
                        }
                    }
                }
            }
        }
    }
}


pub fn sweep_and_prune_neighbor_list(atoms: Res<Atom>, mut neighbor: ResMut<Neighbor>, domain: Res<Domain>, comm: Res<Comm>) {
    let nlocal = atoms.nlocal as usize;
    let total = atoms.pos.len();

    // ── Delay check ──────────────────────────────────────────────────────────
    // Skip the rebuild if no atom (local or ghost) has moved more than half
    // the skin distance since the last build. Ghost positions must be checked
    // too: ghosts are rebuilt every step, so a stale index can point to a
    // different physical atom if we only guard on local displacement.
    if neighbor.last_build_pos.len() == total && total > 0 {
        // Ghost atoms can change identity between builds even if their positions
        // are similar (a different atom enters the boundary region). Check tags
        // to detect this: if any atom's tag changed at the same index, the ghost
        // set changed and we must rebuild.
        let tags_unchanged = atoms.tag
            .iter()
            .zip(neighbor.last_build_tags.iter())
            .all(|(t, lt)| t == lt);

        if tags_unchanged {
            let min_r = atoms.skin[..nlocal].iter().cloned().fold(f64::MAX, f64::min);
            let half_skin = (neighbor.skin_fraction - 1.0) * min_r * 0.5;
            let threshold_sq = half_skin * half_skin;

            let needs_rebuild = atoms.pos
                .iter()
                .zip(neighbor.last_build_pos.iter())
                .any(|(pos, last)| (pos - last).norm_squared() > threshold_sq);

            if !needs_rebuild {
                neighbor.steps_since_build += 1;
                return;
            }
        }
    }

    // ── Rebuild ──────────────────────────────────────────────────────────────
    neighbor.last_build_pos.clear();
    neighbor.last_build_pos.extend_from_slice(&atoms.pos);
    neighbor.last_build_tags.clear();
    neighbor.last_build_tags.extend_from_slice(&atoms.tag);
    neighbor.steps_since_build = 0;

    neighbor.sweep_and_prune.clear();
    neighbor.neighbor_list.clear();

    // println!("sweep and prune {} {}", atoms.pos.len(), comm.rank);
    for j in 0..(atoms.pos.len()) { 
        neighbor.sweep_and_prune.push((j, atoms.pos[j].x));
         
    } 

    neighbor.sweep_and_prune.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());


    for i in 0..neighbor.sweep_and_prune.len() {
        let index = neighbor.sweep_and_prune[i].0;
        let pos = atoms.pos[index];
        let r = atoms.skin[index];

        for j in (i+1)..neighbor.sweep_and_prune.len() {
            if (neighbor.sweep_and_prune[j].1 - pos.x) > (r * 2.0 * neighbor.skin_fraction) {
                break;
            }
            let index2 = neighbor.sweep_and_prune[j].0;
            if atoms.tag[index] == atoms.tag[index2] || (atoms.is_ghost[index] == true &&  atoms.is_ghost[index2] == true) {
                continue;
            }
            let p1 = atoms.pos[index];
            let p2 = atoms.pos[index2];
            let r1 = atoms.skin[index];
            let r2 = atoms.skin[index2];
            let position_difference = p2 - p1;
            let distance = position_difference.norm();  

            if distance < (r1 + r2)*neighbor.skin_fraction {
                neighbor.neighbor_list.push((index, index2));
            }
        }
    }
}



pub fn brute_force_neighbor_list(atoms: Res<Atom>, mut neighbor: ResMut<Neighbor>) {
    neighbor.neighbor_list.clear();

    for i in 0..(atoms.pos.len() - atoms.nghost as usize) {
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
                neighbor.neighbor_list.push((i, j));
            }
        }
    }
}

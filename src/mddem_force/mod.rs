

use std::{any::TypeId};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use crate::{dem_atom::DemAtom, mddem_atom::Atom, mddem_neighbor::Neighbor};

pub struct ForcePlugin;

impl Plugin for ForcePlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Force::new())
            .add_update_system(hertz_normal_force, ScheduleSet::Force);
    }
}

pub struct Force {
    pub _skin_fraction: f64,
}


impl Force {
    pub fn new() -> Self {
        Force {
            _skin_fraction: 1.0,
        }
    }
}



// pub fn read_input(input: Res<Input>, comm: Res<Comm>,) {
//     let commands = &input.commands;
//     for c in commands.iter() {
//         let values = c.split_whitespace().collect::<Vec<&str>>();

//         if values.len() > 0 {
//             match values[0] {
//                 // "neighbor" => {
//                 //     if comm.rank == 0 {
//                 //         println!("Comm: {}", c);
//                 //     }

//                 //     neighbor.skin_fraction = values[1].parse::<f64>().unwrap();
//                 // }
//                 _ => {}
//             }
//         }
//     }
// }

// pub fn setup(mut neighbor: ResMut<Neighbor>) {

// }




pub fn hertz_normal_force(mut atoms: ResMut<Atom>, neighbor: Res<Neighbor>) {
    for ((i_index,j_index), neighbor) in &neighbor.neighbor_list_map {
        let i = *i_index;
        let j = *j_index;

        let mut force_fraction = 1.0;
        if neighbor.add_half_force {
            force_fraction = 0.5;
        }

        let p1 = atoms.pos[i];
        let p2 = atoms.pos[j];
        let v1 = atoms.velocity[i];
        let v2 = atoms.velocity[j];

        let mut r1 = 0.0;
        let mut r2 = 0.0;
        let mut beta = 0.0;

        let mut poisson_ratio1 = 0.0;
        let mut poisson_ratio2 = 0.0;
        let mut youngs_mod1 = 0.0;
        let mut youngs_mod2 = 0.0;


        if let Some(dem_atoms_option) = atoms.added.get(&TypeId::of::<DemAtom>()) {
            let mut dem_atom_binder = dem_atoms_option.borrow_mut();
            let dem_atom = dem_atom_binder.downcast_mut::<DemAtom>().unwrap();

            r1 = dem_atom.radius[i];
            r2 = dem_atom.radius[j];
            beta = dem_atom.beta;
            poisson_ratio1 = dem_atom.poisson_ratio[i];
            poisson_ratio2 = dem_atom.poisson_ratio[j];

            youngs_mod1 = dem_atom.youngs_mod[i];
            youngs_mod2 = dem_atom.youngs_mod[j];
        }

        
        

        

        let position_difference = p2 - p1;

        let distance = position_difference.norm();

        if distance == 0.0 {
            println!("tags {} {}", atoms.tag[i], atoms.tag[j]);
        }

        if distance < r1 + r2 { 

            

            atoms.is_collision[i] = true;
            atoms.is_collision[j] = true;

            let normalized_delta = position_difference / distance;
            // println!("pos dif {} dis {}", position_difference, distance);
            let distance_delta = (r1 + r2) - distance;
            // if distance_delta > 0.000001 {
            //     println!("{}", distance_delta)
            // }

            let effective_radius = 1.0 / (1.0 / r1 + 1.0 / r2);

            let effective_youngs = 1.0
                / ((1.0 - poisson_ratio1 * poisson_ratio2)
                    / youngs_mod1
                    + (1.0 - poisson_ratio1 * poisson_ratio2)
                        / youngs_mod2);
            let normal_force = 4.0 / 3.0
                * effective_youngs
                * effective_radius.sqrt()
                * distance_delta.powf(3.0 / 2.0);

            let contact_stiffness =
                2.0 * effective_youngs * (effective_radius * distance_delta).sqrt();
            let reduced_mass =
                atoms.mass[i] * atoms.mass[j] / (atoms.mass[i] + atoms.mass[j]);

            let delta_veloctiy = v2 - v1;
            let f_dot = normalized_delta.dot(&delta_veloctiy);

            // println!("fdot {} {} ", f_dot, normalized_delta);
            let v_r_n = f_dot * normalized_delta;


            // println!("{} {} {} {}", atoms.beta, contact_stiffness, reduced_mass, v_r_n);
            let dissipation_force = 2.0
                * 0.91287092917
                * beta
                * (contact_stiffness * reduced_mass).sqrt()
                * v_r_n.norm()
                * v_r_n.dot(&normalized_delta).signum();

            // println!("{} {}", normal_force, dissipation_force);
            atoms.force[i] -= (normal_force - dissipation_force) * normalized_delta * force_fraction;
            atoms.force[j] += (normal_force - dissipation_force) * normalized_delta * force_fraction;
        }
    }
}

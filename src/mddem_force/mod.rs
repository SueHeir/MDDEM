

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use crate::{mddem_atom::Atom, mddem_neighbor::Neighbor};

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
    for ((i_index,j_index), _neighbor) in &neighbor.neighbor_list_map {
        let i = *i_index;
        let j = *j_index;
        let p1 = atoms.pos[i];
        let p2 = atoms.pos[j];
        let v1 = atoms.velocity[i];
        let v2 = atoms.velocity[j];
        let r1 = atoms.radius[i];
        let r2 = atoms.radius[j];

        

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
            let distance_delta = (atoms.radius[i] + atoms.radius[j]) - distance;

            let effective_radius = 1.0 / (1.0 / atoms.radius[i] + 1.0 / atoms.radius[j]);

            let effective_youngs = 1.0
                / ((1.0 - atoms.poisson_ratio[i] * atoms.poisson_ratio[i])
                    / atoms.youngs_mod[i]
                    + (1.0 - atoms.poisson_ratio[j] * atoms.poisson_ratio[j])
                        / atoms.youngs_mod[j]);
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
                * atoms.beta
                * (contact_stiffness * reduced_mass).sqrt()
                * v_r_n.norm()
                * v_r_n.dot(&normalized_delta).signum();

            // println!("{} {}", normal_force, dissipation_force);
            atoms.force[i] -= (normal_force - dissipation_force) * normalized_delta;
            atoms.force[j] += (normal_force - dissipation_force) * normalized_delta;
        }
    }
}

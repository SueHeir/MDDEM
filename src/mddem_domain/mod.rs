use std::process::exit;

use nalgebra::Vector3;
use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use crate::{mddem_atom::Atom, mddem_communication::Comm, mddem_input::Input};

pub struct DomainPlugin;

impl Plugin for DomainPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Domain::new())
            .add_setup_system(read_input, ScheduleSet::Setup)
            .add_update_system(pbc, ScheduleSet::PreExchange);
    }
}



pub struct Domain {
    pub boundaries_low: Vector3<f64>,
    pub boundaries_high: Vector3<f64>,
    pub sub_domain_low: Vector3<f64>,
    pub sub_domain_high: Vector3<f64>,
    pub sub_length: Vector3<f64>,
    pub volume: f64,
    pub size: Vector3<f64>,
    pub is_periodic: Vector3<bool>,
}

impl Domain {
    pub fn new() -> Self {
        Domain {
            boundaries_high: Vector3::new(1.0, 1.0, 1.0),
            boundaries_low: Vector3::new(0.0, 0.0, 0.0),
            sub_domain_low: Vector3::new(0.0, 0.0, 0.0),
            sub_domain_high: Vector3::new(1.0, 1.0, 1.0),
            sub_length: Vector3::new(1.0, 1.0, 1.0),

            size: Vector3::new(1.0, 1.0, 1.0),
            is_periodic: Vector3::new(false, false, false),
            volume: 1.0,
        }
    }
}




pub(crate) fn read_input(input: Res<Input>, comm: Res<Comm>, mut domain: ResMut<Domain>) {
    let commands = &input.commands;
    for c in commands.iter() {
        let values = c.split_whitespace().collect::<Vec<&str>>();

        if values.len() > 0 {
            match values[0] {
                "domain" => {
                    if comm.rank == 0 {
                        println!("Domain: {}", c);
                    }

                    if values.len() != 7 {
                        if comm.rank == 0 {
                            println!("Domain: Please fill out domain command with x_low x_high y_low y_high z_low z_high");
                        }
                        exit(1);
                    }

                    domain.boundaries_low.x = values[1].parse::<f64>().unwrap();
                    domain.boundaries_low.y = values[3].parse::<f64>().unwrap();
                    domain.boundaries_low.z = values[5].parse::<f64>().unwrap();

                    domain.boundaries_high.x = values[2].parse::<f64>().unwrap();
                    domain.boundaries_high.y = values[4].parse::<f64>().unwrap();
                    domain.boundaries_high.z = values[6].parse::<f64>().unwrap();

                    domain.size = domain.boundaries_high - domain.boundaries_low;

                    //The sub domain is the location this specific processor is responsible for.
                    let delta_x = (domain.boundaries_high[0] - domain.boundaries_low[0])
                        / comm.processor_decomposition[0] as f64;
                    let delta_y = (domain.boundaries_high[1] - domain.boundaries_low[1])
                        / comm.processor_decomposition[1] as f64;
                    let delta_z = (domain.boundaries_high[2] - domain.boundaries_low[2])
                        / comm.processor_decomposition[2] as f64;

                    domain.sub_domain_low.x =
                        domain.boundaries_low.x + delta_x * comm.processor_position.x as f64;
                    domain.sub_domain_low.y =
                        domain.boundaries_low.y + delta_y * comm.processor_position.y as f64;
                    domain.sub_domain_low.z =
                        domain.boundaries_low.z + delta_z * comm.processor_position.z as f64;

                    domain.sub_domain_high.x =
                        domain.boundaries_low.x + delta_x * (1 + comm.processor_position.x) as f64;
                    domain.sub_domain_high.y =
                        domain.boundaries_low.y + delta_y * (1 + comm.processor_position.y) as f64;
                    domain.sub_domain_high.z =
                        domain.boundaries_low.z + delta_z * (1 + comm.processor_position.z) as f64;

                    domain.sub_length = Vector3::new(delta_x, delta_y, delta_z);

                    // println!("Comm Rank: {0}, subdomain_low: {1} {2} {3} subdomain_high {4} {5} {6}", comm.rank, domain.sub_domain_low.x,domain.sub_domain_low.y,domain.sub_domain_low.z, domain.sub_domain_high.x,domain.sub_domain_high.y,domain.sub_domain_high.z);
                }

                "periodic" => {
                    if comm.rank == 0 {
                        println!("Domain: {}", c);
                    }

                    if values.len() != 4 {
                        if comm.rank == 0 {
                            println!("Domain: Please fill out periodic command as perodic    p p p  or perodic    n n n");
                        }
                        exit(1);
                    }
                    if values[1] == "p" {
                        domain.is_periodic.x = true;
                    }
                    if values[2] == "p" {
                        domain.is_periodic.y = true;
                    }
                    if values[3] == "p" {
                        domain.is_periodic.z = true;
                    }
                }

                _ => {}
            }
        }
    }
}

pub fn pbc(mut atoms: ResMut<Atom>, domain: Res<Domain>) {
    for i in (0..atoms.radius.len()).rev() {
        if domain.is_periodic.x {
            while atoms.pos[i].x < domain.boundaries_low.x {
                atoms.pos[i].x += domain.size.x
            }
            while atoms.pos[i].x >= domain.boundaries_high.x {
                atoms.pos[i].x -= domain.size.x
            }
        } else {
            if atoms.pos[i].x < domain.boundaries_low.x
                || atoms.pos[i].x >= domain.boundaries_high.x
            {
                println!("xdel");
                atoms.delete(i);
                continue;
            }
        }
        if domain.is_periodic.y {
            while atoms.pos[i].y < domain.boundaries_low.y {
                atoms.pos[i].y += domain.size.y
            }
            while atoms.pos[i].y >= domain.boundaries_high.y {
                atoms.pos[i].y -= domain.size.y
            }
        } else {
            if atoms.pos[i].y < domain.boundaries_low.y
                || atoms.pos[i].y >= domain.boundaries_high.y
            {
                atoms.delete(i);
                println!("ydel");
                continue;
            }
        }
        if domain.is_periodic.z {
            while atoms.pos[i].z < domain.boundaries_low.z {
                atoms.pos[i].z += domain.size.z
            }
            while atoms.pos[i].z >= domain.boundaries_high.z {
                atoms.pos[i].z -= domain.size.z
            }
        } else {
            if atoms.pos[i].z < domain.boundaries_low.z
                || atoms.pos[i].z >= domain.boundaries_high.z
            {
                atoms.delete(i);
                println!("zdel");
                continue;
            }
        }
    }
}

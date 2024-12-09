use std::{
    f64::consts::PI,
    ops::{Index, IndexMut},
};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use crate::{mddem_communication::Comm, mddem_domain::Domain, mddem_input::Input};

use mpi::traits::{CommunicatorCollectives, Equivalence};
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use rand::{thread_rng, Rng};






pub struct AtomPlugin;

impl Plugin for AtomPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Atom::new())
            .add_setup_system(read_input, ScheduleSet::Setup)
            .add_setup_system(calculate_delta_time, ScheduleSet::PreInitalIntegration)
            .add_update_system(remove_ghost_atoms, ScheduleSet::PostInitalIntegration)
            .add_update_system(zero_all_forces, ScheduleSet::PostInitalIntegration);
    }
}

#[derive(Equivalence, Debug, Clone, Copy)]
pub struct Vector3f64MPI {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Index<&'_ usize> for Vector3f64MPI {
    type Output = f64;
    fn index(&self, s: &usize) -> &f64 {
        match s {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("unknown field: {}", s),
        }
    }
}

impl IndexMut<&'_ usize> for Vector3f64MPI {
    fn index_mut(&mut self, s: &usize) -> &mut f64 {
        match s {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("unknown field: {}", s),
        }
    }
}

impl Vector3f64MPI {
    fn new(vec: Vector3<f64>) -> Vector3f64MPI {
        Vector3f64MPI {
            x: vec.x,
            y: vec.y,
            z: vec.z,
        }
    }

    fn to_vector3(self) -> Vector3<f64> {
        let vec = Vector3::new(self.x, self.y, self.z);
        vec
    }
}

#[derive(Equivalence, Debug, Clone, Copy)]
pub struct Quaternionf64MPI {
    i: f64,
    j: f64,
    k: f64,
    w: f64,
}

impl Quaternionf64MPI {
    fn new(vec: UnitQuaternion<f64>) -> Quaternionf64MPI {
        Quaternionf64MPI {
            i: vec.i,
            j: vec.j,
            k: vec.k,
            w: vec.w,
        }
    }

    fn to_quat(self) -> UnitQuaternion<f64> {
        let vec = UnitQuaternion::from_quaternion(Quaternion::new(self.w, self.i, self.j, self.k));
        vec
    }
}


pub(crate) struct Atom {
    pub natoms: u64,
    pub nlocal: u32,
    pub nghost: u32,

    pub dt: f64,
    pub restitution_coefficient: f64,
    pub beta: f64,

    pub tag: Vec<u32>,
    pub origin_index: Vec<i32>,
    pub is_ghost: Vec<bool>,
    pub is_collision: Vec<bool>,

    pub pos: Vec<Vector3<f64>>,
    pub velocity: Vec<Vector3<f64>>,
    pub quaterion: Vec<UnitQuaternion<f64>>,
    pub omega: Vec<Vector3<f64>>,
    pub angular_momentum: Vec<Vector3<f64>>,
    pub torque: Vec<Vector3<f64>>,
    pub force: Vec<Vector3<f64>>,

    pub radius: Vec<f64>,
    pub mass: Vec<f64>,
    pub density: Vec<f64>,

    pub youngs_mod: Vec<f64>,
    pub poisson_ratio: Vec<f64>,
}

#[derive(Equivalence, Debug, Clone, Copy)]
pub(crate) struct AtomMPI {
    pub tag: u32,
    pub origin_index: u32,
    pub pos: Vector3f64MPI,
    pub velocity: Vector3f64MPI,
    pub quaterion: Quaternionf64MPI,
    pub omega: Vector3f64MPI,
    pub angular_momentum: Vector3f64MPI,
    pub torque: Vector3f64MPI,
    pub force: Vector3f64MPI,

    pub radius: f64,
    pub mass: f64,
    pub density: f64,

    pub youngs_mod: f64,
    pub poisson_ratio: f64,
}


#[derive(Equivalence, Debug, Clone, Copy)]
pub(crate) struct ForceMPI {
    pub tag: u32,
    pub origin_index: u32,
    pub torque: Vector3f64MPI,
    pub force: Vector3f64MPI,
}


impl Atom {
    pub fn new() -> Self {
        Atom {
            natoms: 0,
            nlocal: 0,
            nghost: 0,
            dt: 1.0,
            restitution_coefficient: 1.0,
            beta: 1.0,
            tag: Vec::new(),
            origin_index: Vec::new(),
            is_ghost: Vec::new(),
            is_collision: Vec::new(),
            pos: Vec::new(),
            velocity: Vec::new(),
            quaterion: Vec::new(),
            omega: Vec::new(),
            angular_momentum: Vec::new(),
            torque: Vec::new(),
            force: Vec::new(),
            radius: Vec::new(),
            mass: Vec::new(),
            density: Vec::new(),
            youngs_mod: Vec::new(),
            poisson_ratio: Vec::new(),
        }
    }

    pub fn get_max_tag(&self) -> u32 {
        let mut max_tag = 0;
        for t in &self.tag {
            max_tag = max_tag.max(*t);
        }
        return max_tag;
    }

    pub fn get_atom_mpi(&mut self, i: usize) -> AtomMPI {
        let atom_mpi = AtomMPI {
            tag: self.tag.remove(i),
            origin_index: 0,
            pos: Vector3f64MPI::new(self.pos.remove(i)),
            velocity: Vector3f64MPI::new(self.velocity.remove(i)),
            quaterion: Quaternionf64MPI::new(self.quaterion.remove(i)),
            omega: Vector3f64MPI::new(self.omega.remove(i)),
            angular_momentum: Vector3f64MPI::new(self.angular_momentum.remove(i)),
            torque: Vector3f64MPI::new(self.torque.remove(i)),
            force: Vector3f64MPI::new(self.force.remove(i)),
            radius: self.radius.remove(i),
            mass: self.mass.remove(i),
            density: self.density.remove(i),
            youngs_mod: self.youngs_mod.remove(i),
            poisson_ratio: self.poisson_ratio.remove(i),
        };

        self.origin_index.remove(i);
        self.is_collision.remove(i);
        self.is_ghost.remove(i);
        return atom_mpi;
    }

    pub fn copy_atom_mpi(&mut self, i: usize) -> AtomMPI {
        let atom_mpi = AtomMPI {
            tag: self.tag[i].clone(),
            origin_index: i as u32,
            pos: Vector3f64MPI::new(self.pos[i].clone()),
            velocity: Vector3f64MPI::new(self.velocity[i].clone()),
            quaterion: Quaternionf64MPI::new(self.quaterion[i].clone()),
            omega: Vector3f64MPI::new(self.omega[i].clone()),
            angular_momentum: Vector3f64MPI::new(self.angular_momentum[i].clone()),
            torque: Vector3f64MPI::new(self.torque[i].clone()),
            force: Vector3f64MPI::new(self.force[i].clone()),
            radius: self.radius[i].clone(),
            mass: self.mass[i].clone(),
            density: self.density[i].clone(),
            youngs_mod: self.youngs_mod[i].clone(),
            poisson_ratio: self.poisson_ratio[i].clone(),
        };
        return atom_mpi;
    }

    pub fn add_atom_from_atom_mpi(&mut self, atom: AtomMPI, is_ghost: bool) {
        self.tag.push(atom.tag);
        self.origin_index.push(atom.origin_index as i32);
        self.pos.push(atom.pos.to_vector3());
        self.velocity.push(atom.velocity.to_vector3());
        self.quaterion.push(atom.quaterion.to_quat());
        self.omega.push(atom.omega.to_vector3());
        self.angular_momentum
            .push(atom.angular_momentum.to_vector3());
        self.torque.push(atom.torque.to_vector3());
        self.force.push(atom.force.to_vector3());
        self.radius.push(atom.radius);
        self.mass.push(atom.mass);
        self.density.push(atom.density);
        self.youngs_mod.push(atom.youngs_mod);
        self.poisson_ratio.push(atom.poisson_ratio);
        
        self.is_collision.push(false);
        self.is_ghost.push(is_ghost);

    }

    //gets data from ghost to reverse send force data back to real particle
    pub fn get_force_data(&mut self, i: usize) -> ForceMPI {
        if !self.is_ghost[i] {
            // println!("get_force_data from no ghost");
            panic!();
        }
        let force_mpi = ForceMPI {
            tag: self.tag[i].clone(),
            origin_index: self.origin_index[i] as u32,
            torque: Vector3f64MPI::new(self.torque[i].clone()),
            force: Vector3f64MPI::new(self.force[i].clone()),
        };
        force_mpi
    }

    pub fn apply_force_data(&mut self, force_mpi: ForceMPI, _rank: i32, _swap: i32,
    _dim: i32) {
        let i = force_mpi.origin_index as usize;
        if force_mpi.tag != self.tag[i] {
            // println!("apply force wrong tag");
            // println!("rank {} swap {} dim {}", rank, swap, dim);
            panic!();
        }
        // } else {
        //     // println!("apply force correct tag");
        // }
        self.force[i] += force_mpi.force.to_vector3();
        self.torque[i] += force_mpi.torque.to_vector3();
    }

    pub fn delete(&mut self, i: usize) {
        self.tag.remove(i);
        self.origin_index.remove(i);
        self.pos.remove(i);
        self.velocity.remove(i);
        self.quaterion.remove(i);
        self.omega.remove(i);
        self.angular_momentum.remove(i);
        self.torque.remove(i);
        self.force.remove(i);
        self.radius.remove(i);
        self.mass.remove(i);
        self.density.remove(i);
        self.youngs_mod.remove(i);
        self.poisson_ratio.remove(i);
        self.is_collision.remove(i);
        self.is_ghost.remove(i);
    }

    // pub fn get_atom_buff(&mut self, i: usize)-> Vec<f64> {
    //     let mut buff = Vec::new();
    //     buff.push(self.tag.remove(i) as f64);

    //     let pos = self.pos.remove(i);
    //     buff.push(pos.x);
    //     buff.push(pos.y);
    //     buff.push(pos.z);

    //     let vel = self.velocity.remove(i);
    //     buff.push(vel.x);
    //     buff.push(vel.y);
    //     buff.push(vel.z);

    //     let quat = self.quaterion.remove(i);
    //     buff.push(quat.w);
    //     buff.push(quat.i);
    //     buff.push(quat.j);
    //     buff.push(quat.k);

    //     let omega = self.omega.remove(i);
    //     buff.push(omega.x);
    //     buff.push(omega.y);
    //     buff.push(omega.z);

    //     let angular_momentum = self.angular_momentum.remove(i);
    //     buff.push(angular_momentum.x);
    //     buff.push(angular_momentum.y);
    //     buff.push(angular_momentum.z);

    //     let torque = self.torque.remove(i);
    //     buff.push(torque.x);
    //     buff.push(torque.y);
    //     buff.push(torque.z);

    //     let force = self.force.remove(i);
    //     buff.push(force.x);
    //     buff.push(force.y);
    //     buff.push(force.z);

    //     buff.push(self.radius.remove(i));
    //     buff.push(self.mass.remove(i));
    //     buff.push(self.density.remove(i));
    //     buff.push(self.youngs_mod.remove(i));
    //     buff.push(self.poisson_ratio.remove(i));

    //     return buff;

    // }

    // pub fn add_atom_from_buff(&mut self, mut buff: Vec<f64>) -> Vec<f64> {
    //     self.tag.push(buff.remove(0) as u32);
    //     self.pos.push(Vector3::new(buff.remove(0),buff.remove(0),buff.remove(0)));
    //     self.velocity.push(Vector3::new(buff.remove(0),buff.remove(0),buff.remove(0)));
    //     self.quaterion.push(UnitQuaternion::from_quaternion(Quaternion::new(buff.remove(0),buff.remove(0),buff.remove(0),buff.remove(0))));
    //     self.omega.push(Vector3::new(buff.remove(0),buff.remove(0),buff.remove(0)));
    //     self.angular_momentum.push(Vector3::new(buff.remove(0),buff.remove(0),buff.remove(0)));
    //     self.torque.push(Vector3::new(buff.remove(0),buff.remove(0),buff.remove(0)));
    //     self.force.push(Vector3::new(buff.remove(0),buff.remove(0),buff.remove(0)));
    //     self.radius.push(buff.remove(0));
    //     self.mass.push(buff.remove(0));
    //     self.density.push(buff.remove(0));
    //     self.youngs_mod.push(buff.remove(0));
    //     self.poisson_ratio.push(buff.remove(0));

    //     return buff;
    // }
}

pub fn read_input(input: Res<Input>, scheduler_manager: Res<SchedulerManager>, comm: Res<Comm>, domain: Res<Domain>, mut atom: ResMut<Atom>) {
    let commands = &input.current_commands[scheduler_manager.index];
    let mut rng = thread_rng();
    for c in commands.iter() {
        let values = c.split_whitespace().collect::<Vec<&str>>();

        if values.len() > 0 {
            match values[0] {
                "randomparticleinsert" => {
                    //TODO Make a parallelized version of generating particles
                    if comm.rank == 0 {
                        println!("Atom: {}", c);

                        let particles_to_add: u32 = values[1].parse::<u32>().unwrap();
                        let radius: f64 = values[2].parse::<f64>().unwrap();
                        let density: f64 = values[3].parse::<f64>().unwrap();

                        let youngs_mod: f64 = values[4].parse::<f64>().unwrap();
                        let poisson_ratio: f64 = values[5].parse::<f64>().unwrap();
                        let mut max_tag = atom.get_max_tag();
                        
                        let mut count = 0;

                        while count < particles_to_add {
                            let x =
                                rng.gen_range(domain.boundaries_low[0] + radius..domain.boundaries_high[0] - radius);
                            let y =
                                rng.gen_range(domain.boundaries_low[1] + radius..domain.boundaries_high[1] - radius);
                            let z =
                                rng.gen_range(domain.boundaries_low[2] + radius..domain.boundaries_high[2] - radius);

                            let pos = Vector3::<f64>::new(x, y, z);

                            let mut no_overlap = true;
                            for i in 0..atom.radius.len() {
                                let difference = pos - atom.pos[i];
                                let distance = difference.norm();
                                if distance <= (radius + atom.radius[i]) * 1.1 {
                                    no_overlap = false;
                                }
                            }

                            if no_overlap {
                                count += 1;
                                atom.natoms += 1;
                                atom.nlocal += 1;
                                atom.tag.push(max_tag);
                                atom.origin_index.push(0);
                                atom.is_collision.push(false);
                                atom.is_ghost.push(false);
                                max_tag += 1;
                                atom.pos.push(pos);
                                atom.velocity.push(Vector3::<f64>::zeros());
                                atom.quaterion.push(UnitQuaternion::identity());
                                atom.omega.push(Vector3::<f64>::zeros());
                                atom.angular_momentum.push(Vector3::<f64>::zeros());
                                atom.torque.push(Vector3::<f64>::zeros());
                                atom.force.push(Vector3::<f64>::zeros());
                                atom.radius.push(radius);
                                atom.mass.push(density * 4.0 / 3.0 * PI * radius.powi(3));
                                atom.density.push(density);
                                atom.youngs_mod.push(youngs_mod);
                                atom.poisson_ratio.push(poisson_ratio);
                            }
                        }
                        // for _i in 0..particles_to_add {
                        //     let x =
                        //         rng.gen_range(domain.boundaries_low[0]..domain.boundaries_high[0]);
                        //     let y =
                        //         rng.gen_range(domain.boundaries_low[1]..domain.boundaries_high[1]);
                        //     let z =
                        //         rng.gen_range(domain.boundaries_low[2]..domain.boundaries_high[2]);
                        //     let pos = Vector3::<f64>::new(x, y, z);

                        //     atom.natoms += 1;
                        //     atom.nlocal += 1;
                        //     atom.tag.push(max_tag);
                        //     max_tag += 1;
                        //     atom.pos.push(pos);
                        //     atom.velocity.push(Vector3::<f64>::zeros());
                        //     atom.quaterion.push(UnitQuaternion::identity());
                        //     atom.omega.push(Vector3::<f64>::zeros());
                        //     atom.angular_momentum.push(Vector3::<f64>::zeros());
                        //     atom.torque.push(Vector3::<f64>::zeros());
                        //     atom.force.push(Vector3::<f64>::zeros());
                        //     atom.radius.push(radius);
                        //     atom.mass.push(density * 4.0 / 3.0 * PI * radius.powi(3));
                        //     atom.density.push(density);
                        //     atom.youngs_mod.push(youngs_mod);
                        //     atom.poisson_ratio.push(poisson_ratio);
                        // }
                    }
                }

                "randomparticlevelocity" => {
                    let rand_vel: f64 = values[1].parse::<f64>().unwrap();

                    for v in &mut atom.velocity {
                        v.x = rng.gen_range(-rand_vel..rand_vel);
                        v.y = rng.gen_range(-rand_vel..rand_vel);
                        v.z = rng.gen_range(-rand_vel..rand_vel);
                    }
                }

                "dampening" => {
                    let restitution_coefficient: f64 = values[1].parse::<f64>().unwrap();
                    atom.restitution_coefficient = restitution_coefficient;
                    let log_e = atom.restitution_coefficient.ln();
                    let beta = -log_e / (PI * PI + log_e * log_e).sqrt();
                    atom.beta = beta;
                }



                

                _ => {}
            }
        }
    }
}

fn calculate_delta_time(comm: Res<Comm>, mut atoms: ResMut<Atom>) {
    //Checks each particles Size for the smallest delta time the simulation should use
    let mut dt: f64 = 0.001;
    for i in 0..atoms.radius.len() {
        let g = atoms.youngs_mod[i] / (2.0 * (1.0 + atoms.poisson_ratio[i]));
        let alpha = 0.1631 * atoms.poisson_ratio[i] + 0.876605;
        let delta = PI * atoms.radius[i] / alpha * (atoms.density[i] / g).sqrt();
        dt = delta.min(dt);
    }

    let u = vec![dt; comm.size as usize];
    let mut v = vec![0.0; comm.size as usize];

    comm.world.all_to_all_into(&u[..], &mut v[..]);

    dt = v.into_iter().reduce(f64::min).unwrap();

    println!("Useing {} for delta time", dt * 0.3);
    atoms.dt = dt * 0.3;
}

fn print_all_atom_position(comm: Res<Comm>, atoms: Res<Atom>) {
    for i in 0..atoms.radius.len() {
        println!(
            "{0} pos: {1:?} vel: {2:?}",
            comm.rank, atoms.pos[i], atoms.velocity[i]
        );
    }
}

fn remove_ghost_atoms(mut atoms: ResMut<Atom>) {

    let total = atoms.radius.len();
    for i in (total - atoms.nghost as usize..total).rev() {
        if !atoms.is_ghost[i] {
            println!("removed non ghost atom");
        }
        atoms.delete(i);
    }
}


fn zero_all_forces(mut atoms: ResMut<Atom>) {
    for i in 0..atoms.radius.len() {
        atoms.is_collision[i] = false;
        atoms.force[i] = Vector3::zeros();
        atoms.torque[i] = Vector3::zeros();
    }
}

use std::{any::TypeId, cell::RefCell, f64::consts::PI};
use  downcast::{downcast, Any};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use mpi::traits::CommunicatorCollectives;

use crate::{ mddem_atom::{Atom, AtomAdded, AtomPlugin}, mddem_communication::Comm, mddem_input::Input};


pub struct DemAtomPlugin;

impl Plugin for DemAtomPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(AtomPlugin);

        if let Some(atom_option) = app.get_mut_resource(TypeId::of::<Atom>()) {
            let mut atom_binder = atom_option.borrow_mut();
            let atom = atom_binder.downcast_mut::<Atom>().unwrap();
            atom.added.insert(TypeId::of::<DemAtom>(), RefCell::new(Box::new(DemAtom::new())));
        } else {
            panic!("You need the atom resource to use DemAtomPlugin");
        }

        app.add_setup_system(read_input, ScheduleSetupSet::Setup)
            .add_setup_system(calculate_delta_time, ScheduleSetupSet::PostSetup);
    }
}


pub struct DemAtom {
    pub restitution_coefficient: f64,
    pub beta: f64,

    pub radius: Vec<f64>,
    pub density: Vec<f64>,

    pub youngs_mod: Vec<f64>,
    pub poisson_ratio: Vec<f64>,
}

impl DemAtom {
    pub fn new() -> Self {
        DemAtom {
            restitution_coefficient: 1.0,
            beta: 1.0,
            radius: Vec::new(),
            density: Vec::new(),
            youngs_mod: Vec::new(),
            poisson_ratio: Vec::new(),
        }
    }
}

impl AtomAdded for DemAtom {

    fn delete(&mut self, i: usize) {
        self.radius.remove(i);
        self.density.remove(i);
        self.youngs_mod.remove(i);
        self.poisson_ratio.remove(i);
    }

    fn get_mpi(&mut self, i: usize) -> Vec<f64> {
        let mut buff = Vec::new();

        buff.push(self.radius.remove(i));
        buff.push(self.density.remove(i));
        buff.push(self.youngs_mod.remove(i));
        buff.push(self.poisson_ratio.remove(i));

        buff
    }

    fn copy_mpi(&mut self, i: usize) -> Vec<f64> {
        let mut buff = Vec::new();

        buff.push(self.radius[i].clone());
        buff.push(self.density[i].clone());
        buff.push(self.youngs_mod[i].clone());
        buff.push(self.poisson_ratio[i].clone());

        buff
    }

    fn set_mpi(&mut self, mut buff: Vec<f64>) -> Vec<f64> {
        self.radius.push(buff.remove(0));
        self.density.push(buff.remove(0));
        self.youngs_mod.push(buff.remove(0));
        self.poisson_ratio.push(buff.remove(0));

        return buff
    }
}


pub fn read_input(input: Res<Input>, scheduler_manager: Res<SchedulerManager>, comm: Res<Comm>, mut atoms: ResMut<Atom>) {
    let commands = &input.current_commands[scheduler_manager.index];
    for c in commands.iter() {
        let values = c.split_whitespace().collect::<Vec<&str>>();

        if values.len() > 0 {
            match values[0] {
                "randomparticleinsert" => {
                    //TODO Make a parallelized version of generating particles
                    if comm.rank == 0 {
                        println!("DemAtom: {}", c);

                        let particles_to_add: u32 = values[1].parse::<u32>().unwrap();
                        let radius: f64 = values[2].parse::<f64>().unwrap();
                        let density: f64 = values[3].parse::<f64>().unwrap();

                        let youngs_mod: f64 = values[4].parse::<f64>().unwrap();
                        let poisson_ratio: f64 = values[5].parse::<f64>().unwrap();
                        
                        let mut count = 0;
                        if let Some(dem_atoms_option) = atoms.added.get(&TypeId::of::<DemAtom>()) {
                            let mut dem_atom_binder = dem_atoms_option.borrow_mut();
                            let dem_atom = dem_atom_binder.downcast_mut::<DemAtom>().unwrap();
                            while count < particles_to_add {
                                count += 1;
                                dem_atom.radius.push(radius);
                                // atom.mass.push(density * 4.0 / 3.0 * PI * radius.powi(3));
                                dem_atom.density.push(density);
                                dem_atom.youngs_mod.push(youngs_mod);
                                dem_atom.poisson_ratio.push(poisson_ratio);
                                
                            }
                        }

                       
                  
                    }
                }

                "dampening" => {
                    let restitution_coefficient: f64 = values[1].parse::<f64>().unwrap();

                    if let Some(dem_atoms_option) = atoms.added.get(&TypeId::of::<DemAtom>()) {
                        let mut dem_atom_binder = dem_atoms_option.borrow_mut();
                        let dem_atom = dem_atom_binder.downcast_mut::<DemAtom>().unwrap(); 
                        
                        dem_atom.restitution_coefficient = restitution_coefficient;
                        let log_e = dem_atom.restitution_coefficient.ln();
                        let beta = -log_e / (PI * PI + log_e * log_e).sqrt();
                        dem_atom.beta = beta;
                    }
                   
                }


                _ => {}
            }
        }
    }
}


fn calculate_delta_time(comm: Res<Comm>, mut atoms: ResMut<Atom>) {
    //Checks each particles Size for the smallest delta time the simulation should use

    let mut dt: f64 = 0.001;
    if let Some(dem_atoms_option) = atoms.added.get(&TypeId::of::<DemAtom>()) {
        let mut dem_atom_binder = dem_atoms_option.borrow_mut();
        let dem_atom = dem_atom_binder.downcast_mut::<DemAtom>().unwrap();
        
       
        for i in 0..dem_atom.radius.len() {
            let g = dem_atom.youngs_mod[i] / (2.0 * (1.0 + dem_atom.poisson_ratio[i]));
            let alpha = 0.1631 * dem_atom.poisson_ratio[i] + 0.876605;
            let delta = PI * dem_atom.radius[i] / alpha * (dem_atom.density[i] / g).sqrt();
            dt = delta.min(dt);
        }
    
        let u = vec![dt; comm.size as usize];
        let mut v = vec![0.0; comm.size as usize];
    
        comm.world.all_to_all_into(&u[..], &mut v[..]);
    
        dt = v.into_iter().reduce(f64::min).unwrap();  
    }

    println!("Useing {} for delta time", dt * 0.3);
    atoms.dt = dt * 0.3;

   
    
}
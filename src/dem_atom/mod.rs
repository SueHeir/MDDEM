use std::{any::{Any, TypeId}, f64::consts::PI};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use mpi::collective::SystemOperation;
use mpi::traits::CommunicatorCollectives;

use crate::{
    mddem_atom::{Atom, AtomData, AtomDataRegistry, AtomPlugin},
    mddem_communication::Comm,
    mddem_input::Input,
};


pub struct DemAtomPlugin;

impl Plugin for DemAtomPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(AtomPlugin);

        // Register DemAtom with the AtomDataRegistry so communication systems
        // pack/unpack it automatically alongside base Atom fields.
        if let Some(registry_option) = app.get_mut_resource(TypeId::of::<AtomDataRegistry>()) {
            let mut registry_binder = registry_option.borrow_mut();
            let registry = registry_binder.downcast_mut::<AtomDataRegistry>().unwrap();
            registry.register(DemAtom::new());
        } else {
            panic!("AtomDataRegistry not found — AtomPlugin must be added first");
        }

        app.add_setup_system(read_input, ScheduleSetupSet::Setup)
            .add_setup_system(calculate_delta_time, ScheduleSetupSet::PostSetup);
    }
}


// ── DemAtom ───────────────────────────────────────────────────────────────────

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

impl AtomData for DemAtom {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn truncate(&mut self, n: usize) {
        self.radius.truncate(n);
        self.density.truncate(n);
        self.youngs_mod.truncate(n);
        self.poisson_ratio.truncate(n);
    }

    fn swap_remove(&mut self, i: usize) {
        self.radius.swap_remove(i);
        self.density.swap_remove(i);
        self.youngs_mod.swap_remove(i);
        self.poisson_ratio.swap_remove(i);
    }

    fn pack(&self, i: usize, buf: &mut Vec<f64>) {
        buf.push(self.radius[i]);
        buf.push(self.density[i]);
        buf.push(self.youngs_mod[i]);
        buf.push(self.poisson_ratio[i]);
    }

    fn unpack(&mut self, buf: &[f64]) -> usize {
        self.radius.push(buf[0]);
        self.density.push(buf[1]);
        self.youngs_mod.push(buf[2]);
        self.poisson_ratio.push(buf[3]);
        4
    }
}


// ── Systems ───────────────────────────────────────────────────────────────────

pub fn read_input(
    input: Res<Input>,
    scheduler_manager: Res<SchedulerManager>,
    comm: Res<Comm>,
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
) {
    let commands = &input.current_commands[scheduler_manager.index];
    for c in commands.iter() {
        let values = c.split_whitespace().collect::<Vec<&str>>();

        if values.len() > 0 {
            match values[0] {
                "randomparticleinsert" => {
                    if comm.rank == 0 {
                        println!("DemAtom: {}", c);

                        let particles_to_add: u32 = values[1].parse::<u32>().unwrap();
                        let radius: f64 = values[2].parse::<f64>().unwrap();
                        let density: f64 = values[3].parse::<f64>().unwrap();
                        let youngs_mod: f64 = values[4].parse::<f64>().unwrap();
                        let poisson_ratio: f64 = values[5].parse::<f64>().unwrap();

                        let mut dem = registry.get_mut::<DemAtom>().unwrap();
                        let mut count = 0;
                        while count < particles_to_add {
                            count += 1;
                            dem.radius.push(radius);
                            dem.density.push(density);
                            dem.youngs_mod.push(youngs_mod);
                            dem.poisson_ratio.push(poisson_ratio);
                        }
                    }
                }

                "dampening" => {
                    let restitution_coefficient: f64 = values[1].parse::<f64>().unwrap();
                    let mut dem = registry.get_mut::<DemAtom>().unwrap();
                    dem.restitution_coefficient = restitution_coefficient;
                    let log_e = dem.restitution_coefficient.ln();
                    dem.beta = -log_e / (PI * PI + log_e * log_e).sqrt();
                }

                _ => {}
            }
        }
    }
}

fn calculate_delta_time(comm: Res<Comm>, mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    let dem = registry.get::<DemAtom>().unwrap();
    let mut dt: f64 = 0.001;

    for i in 0..dem.radius.len() {
        let g = dem.youngs_mod[i] / (2.0 * (1.0 + dem.poisson_ratio[i]));
        let alpha = 0.1631 * dem.poisson_ratio[i] + 0.876605;
        let delta = PI * dem.radius[i] / alpha * (dem.density[i] / g).sqrt();
        dt = delta.min(dt);
    }

    let local_dt = dt;
    comm.world.all_reduce_into(&local_dt, &mut dt, SystemOperation::min());

    println!("Using {} for delta time", dt * 0.05);
    atoms.dt = dt * 0.05;
}

use std::any::TypeId;

use super::{
    atom::Atom,
    comm::Comm,
    input::Input,
    scheduler::{Res, ResMut, ScheduleSet::*, Scheduler},
};

pub fn verlet_app(scheduler: &mut Scheduler) {
    scheduler.add_resource(Verlet::new());
    scheduler.add_setup_system(read_input, Setup);
    scheduler.add_update_system(update_cycle, PreInitalIntegration);
    scheduler.add_update_system(inital_integration, InitalIntegration);
    scheduler.add_update_system(final_integration, FinalIntegration);
}

pub struct Verlet {
    pub cycle_count: u32,
    cycle_remaining: u32,
}
impl Verlet {
    pub fn new() -> Self {
        Verlet {
            cycle_count: 0,
            cycle_remaining: 0,
        }
    }
}

pub fn read_input(input: Res<Input>, comm: Res<Comm>, mut verlet: ResMut<Verlet>) {
    let commands = &input.commands;
    for c in commands.iter() {
        let values = c.split_whitespace().collect::<Vec<&str>>();

        if values.len() > 0 {
            match values[0] {
                "run" => {
                    if comm.rank == 0 {
                        println!("Verlet: {}", c);
                    }
                    verlet.cycle_count = 0;
                    verlet.cycle_remaining = values[1].parse::<u32>().unwrap();
                }

                _ => {}
            }
        }
    }
}

pub fn update_cycle(mut verlet: ResMut<Verlet>) {
    verlet.cycle_count += 1
}

pub fn inital_integration(mut atoms: ResMut<Atom>) {
    for i in 0..atoms.pos.len() {
        let velocity_change = 0.5 * atoms.dt * atoms.force[i] / atoms.mass[i];
        atoms.velocity[i] += velocity_change;
        let position_change = atoms.velocity[i] * atoms.dt;
        atoms.pos[i] += position_change;
    }
}

pub fn final_integration(mut atoms: ResMut<Atom>) {
    for i in 0..atoms.pos.len() {
        let velocity_change = 0.5 * atoms.dt * atoms.force[i] / atoms.mass[i];
        atoms.velocity[i] += velocity_change;
    }
}

pub fn run(scheduler: &mut Scheduler) {
    let mut cycle_remaing = 0;

    //get Data from verlet outside of a scheduled system
    {
        let mut binding = scheduler
            .resources
            .get(&TypeId::of::<Verlet>())
            .unwrap()
            .borrow_mut();
        let verlet = binding.downcast_mut::<Verlet>().unwrap();
        cycle_remaing = verlet.cycle_remaining;
    }

    while cycle_remaing > 0 {
        scheduler.run();
        cycle_remaing -= 1;
    }
}

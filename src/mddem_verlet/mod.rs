use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use crate::{mddem_atom::Atom, mddem_communication::Comm, mddem_input::Input};

pub struct VerletPlugin;

impl Plugin for VerletPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Verlet::new())
            .add_setup_system(read_input, ScheduleSetupSet::Setup)
            .add_update_system(update_cycle, ScheduleSet::PostFinalIntegration)
            .add_update_system(inital_integration, ScheduleSet::InitalIntegration)
            .add_update_system(final_integration, ScheduleSet::FinalIntegration);
    }
}

pub struct Verlet {
    pub total_cycle: usize,
    pub cycle_count: Vec<u32>,
    cycle_remaining: Vec<u32>,
}
impl Verlet {
    pub fn new() -> Self {
        Verlet {
            total_cycle: 0,
            cycle_count: Vec::new(),
            cycle_remaining: Vec::new(),
        }
    }
}

pub fn read_input(input: Res<Input>, scheduler_manager: Res<SchedulerManager>, comm: Res<Comm>, mut verlet: ResMut<Verlet>) {
    
    // Only run this input reader once because we go through all commands
    if scheduler_manager.index != 0 {
        return;
    }

    let commands = &input.all_commands;
    for c in commands.iter() {
        let values = c.split_whitespace().collect::<Vec<&str>>();

        if values.len() > 0 {
            match values[0] {
                "run" => {
                    if comm.rank == 0 {
                        println!("Verlet: {}", c);
                    }
                    verlet.cycle_count.push(0);
                    verlet.cycle_remaining.push(values[1].parse::<u32>().unwrap());
                }

                _ => {}
            }
        }
    }
}

pub fn update_cycle(mut verlet: ResMut<Verlet>, mut scheudler_manager: ResMut<SchedulerManager>) {
    let index = scheudler_manager.index;
    verlet.cycle_count[index] += 1;
    verlet.total_cycle += 1;

    if verlet.cycle_count[index] == verlet.cycle_remaining[index] {
        scheudler_manager.index += 1;
        scheudler_manager.state = SchedulerState::Setup;
        
        if scheudler_manager.index == verlet.cycle_count.len() {
            scheudler_manager.state = SchedulerState::End;
            
        
        }
    }
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
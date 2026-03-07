use serde::{Deserialize, Serialize};
use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use crate::{Config, CommResource};

fn default_steps() -> u32 { 1000 }
fn default_thermo() -> usize { 100 }

#[derive(Serialize, Deserialize, Clone)]
pub struct RunConfig {
    #[serde(default = "default_steps")]
    pub steps: u32,
    #[serde(default = "default_thermo")]
    pub thermo: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig { steps: 1000, thermo: 100 }
    }
}

pub struct RunState {
    pub total_cycle: usize,
    pub cycle_count: Vec<u32>,
    cycle_remaining: Vec<u32>,
}

impl RunState {
    pub fn new() -> Self {
        RunState { total_cycle: 0, cycle_count: Vec::new(), cycle_remaining: Vec::new() }
    }
}

pub struct RunPlugin;

impl Plugin for RunPlugin {
    fn build(&self, app: &mut App) {
        Config::load::<RunConfig>(app, "run");

        app.add_resource(RunState::new())
            .add_setup_system(run_read_input, ScheduleSetupSet::Setup)
            .add_update_system(update_cycle, ScheduleSet::PostFinalIntegration);
    }
}

pub fn run_read_input(config: Res<RunConfig>, scheduler_manager: Res<SchedulerManager>, comm: Res<CommResource>, mut run_state: ResMut<RunState>) {
    if scheduler_manager.index != 0 { return; }
    if comm.rank() == 0 { println!("Run: {} steps", config.steps); }
    run_state.cycle_count.push(0);
    run_state.cycle_remaining.push(config.steps);
}

pub fn update_cycle(mut run_state: ResMut<RunState>, mut scheudler_manager: ResMut<SchedulerManager>) {
    let index = scheudler_manager.index;
    run_state.cycle_count[index] += 1;
    run_state.total_cycle += 1;
    if run_state.cycle_count[index] == run_state.cycle_remaining[index] {
        scheudler_manager.index += 1;
        scheudler_manager.state = SchedulerState::Setup;
        if scheudler_manager.index == run_state.cycle_count.len() {
            scheudler_manager.state = SchedulerState::End;
        }
    }
}

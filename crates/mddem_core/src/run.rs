use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{CommResource, Config};

fn default_steps() -> u32 {
    1000
}
fn default_thermo() -> usize {
    100
}

#[derive(Serialize, Deserialize, Clone)]
pub struct StageConfig {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "default_steps")]
    pub steps: u32,
    #[serde(default = "default_thermo")]
    pub thermo: usize,
    #[serde(default)]
    pub dump_interval: Option<usize>,
    #[serde(default)]
    pub restart_interval: Option<usize>,
    #[serde(default)]
    pub vtp_interval: Option<usize>,
}

impl Default for StageConfig {
    fn default() -> Self {
        StageConfig {
            name: None,
            steps: 1000,
            thermo: 100,
            dump_interval: None,
            restart_interval: None,
            vtp_interval: None,
        }
    }
}

#[derive(Clone)]
pub struct RunConfig {
    pub stages: Vec<StageConfig>,
}

impl RunConfig {
    pub fn current_stage(&self, index: usize) -> &StageConfig {
        &self.stages[index.min(self.stages.len() - 1)]
    }

    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            stages: vec![StageConfig::default()],
        }
    }
}

pub struct RunState {
    pub total_cycle: usize,
    pub cycle_count: Vec<u32>,
    cycle_remaining: Vec<u32>,
}

impl Default for RunState {
    fn default() -> Self {
        Self::new()
    }
}

impl RunState {
    pub fn new() -> Self {
        RunState {
            total_cycle: 0,
            cycle_count: Vec::new(),
            cycle_remaining: Vec::new(),
        }
    }
}

pub struct RunPlugin;

impl Plugin for RunPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# Single-stage run:
[run]
steps = 1000
thermo = 100
# name = "my_stage"          # optional stage name
# dump_interval = 0          # override dump interval for this stage
# restart_interval = 0       # override restart interval for this stage
# vtp_interval = 0           # override VTP interval for this stage

# Multi-stage run (use [[run]] instead of [run]):
# [[run]]
# name = "settling"
# steps = 1000
# thermo = 100
#
# [[run]]
# name = "production"
# steps = 5000
# thermo = 500"#,
        )
    }

    fn build(&self, app: &mut App) {
        let run_config = Config::load_run_config(app);
        app.add_resource(run_config);

        app.add_resource(RunState::new())
            .add_setup_system(run_read_input, ScheduleSetupSet::Setup)
            .add_update_system(update_cycle, ScheduleSet::PostFinalIntegration);
    }
}

pub fn run_read_input(
    config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
    comm: Res<CommResource>,
    mut run_state: ResMut<RunState>,
) {
    let index = scheduler_manager.index;
    if index >= config.num_stages() {
        return;
    }

    let stage = config.current_stage(index);
    let stage_label = stage.name.as_deref().unwrap_or("(unnamed)");

    if comm.rank() == 0 {
        if config.num_stages() > 1 {
            println!(
                "Run stage {} [{}]: {} steps, thermo every {}",
                index, stage_label, stage.steps, stage.thermo
            );
        } else {
            println!("Run: {} steps", stage.steps);
        }
    }
    run_state.cycle_count.push(0);
    run_state.cycle_remaining.push(stage.steps);
}

pub fn update_cycle(
    mut run_state: ResMut<RunState>,
    mut scheudler_manager: ResMut<SchedulerManager>,
    run_config: Res<RunConfig>,
) {
    let index = scheudler_manager.index;
    run_state.cycle_count[index] += 1;
    run_state.total_cycle += 1;
    if run_state.cycle_count[index] == run_state.cycle_remaining[index] {
        scheudler_manager.index += 1;
        scheudler_manager.state = SchedulerState::Setup;
        if scheudler_manager.index >= run_config.num_stages() {
            scheudler_manager.state = SchedulerState::End;
        }
    }
}

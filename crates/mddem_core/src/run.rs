//! Run stages, cycle counting, and multi-stage simulation control.
//!
//! Supports single-stage `[run]` or multi-stage `[[run]]` TOML syntax.
//! Each stage has its own step count, thermo interval, and optional config
//! overrides (e.g. change thermostat temperature between stages).

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{CommResource, Config};
use mddem_app::StageNames;

fn default_steps() -> u32 {
    1000
}
fn default_thermo() -> usize {
    100
}

#[derive(Serialize, Deserialize, Clone)]
/// Per-stage settings: step count, thermo/dump/restart intervals, plus arbitrary overrides.
pub struct StageConfig {
    /// Optional human-readable stage name.
    #[serde(default)]
    pub name: Option<String>,
    /// Number of timesteps to run in this stage.
    #[serde(default = "default_steps")]
    pub steps: u32,
    /// Timestep size. If set to 0.0 (default), DEM auto-computes from Rayleigh time.
    /// Set explicitly for rate-based insertion or when you want a specific dt.
    #[serde(default)]
    pub dt: f64,
    /// Print thermo output every N steps.
    #[serde(default = "default_thermo")]
    pub thermo: usize,
    /// Override dump interval for this stage (None = use global).
    #[serde(default)]
    pub dump_interval: Option<usize>,
    /// Override restart interval for this stage (None = use global).
    #[serde(default)]
    pub restart_interval: Option<usize>,
    /// Override VTP interval for this stage (None = use global).
    #[serde(default)]
    pub vtp_interval: Option<usize>,
    /// Skip this stage entirely (e.g. already completed, load from restart).
    #[serde(default)]
    pub skip: bool,
    /// Write dump + restart files when this stage finishes.
    #[serde(default)]
    pub save_at_end: bool,
    /// Arbitrary config overrides for this stage (e.g. `thermostat.temperature = 1.2`).
    /// Captured via `#[serde(flatten)]` — any keys not matched above land here.
    #[serde(flatten)]
    pub overrides: toml::Table,
}

impl Default for StageConfig {
    fn default() -> Self {
        StageConfig {
            name: None,
            steps: 1000,
            dt: 0.0,
            thermo: 100,
            dump_interval: None,
            restart_interval: None,
            vtp_interval: None,
            skip: false,
            save_at_end: false,
            overrides: toml::Table::new(),
        }
    }
}

#[derive(Clone)]
/// All run stages. Supports single `[run]` or multi-stage `[[run]]` TOML syntax.
pub struct RunConfig {
    pub stages: Vec<StageConfig>,
}

impl RunConfig {
    /// Get the config for stage `index`, clamping to the last stage if out of range.
    pub fn current_stage(&self, index: usize) -> &StageConfig {
        &self.stages[index.min(self.stages.len() - 1)]
    }

    /// Total number of configured run stages.
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

/// Mutable state tracking the current cycle count per stage and total.
pub struct RunState {
    pub total_cycle: usize,
    pub cycle_count: Vec<u32>,
    pub cycle_remaining: Vec<u32>,
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

/// Manages run stages, cycle counting, and scheduler state transitions.
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
# skip = false               # skip this stage entirely
# save_at_end = false        # dump + restart when stage finishes

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
        app.add_resource(StageOverrides {
            table: toml::Table::new(),
        });

        app.add_resource(RunState::new())
            .add_setup_system(set_stage_name, ScheduleSetupSet::PreSetup)
            .add_setup_system(run_read_input, ScheduleSetupSet::Setup)
            .add_update_system(update_cycle.label("update_cycle"), ScheduleSet::PostFinalIntegration);

        // If StageAdvancePlugin registered StageNames, add validation
        if app
            .get_resource_ref::<StageNames>()
            .is_some()
        {
            app.add_setup_system(
                validate_stages.run_if(first_stage_only()),
                ScheduleSetupSet::PreSetup,
            );
        }
    }
}

/// Setup system: initialize cycle counters for the current stage and print run info.
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

    if stage.skip {
        if comm.rank() == 0 {
            println!("Skipping stage {} [{}]", index, stage_label);
        }
        run_state.cycle_count.push(0);
        run_state.cycle_remaining.push(0);
        return;
    }

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

/// Increment cycle counters and advance to the next stage when steps are exhausted.
pub fn update_cycle(
    mut run_state: ResMut<RunState>,
    mut scheduler_manager: ResMut<SchedulerManager>,
    run_config: Res<RunConfig>,
) {
    let index = scheduler_manager.index;
    let remaining = run_state.cycle_remaining[index];

    // Skipped stage (remaining == 0): advance immediately without running physics.
    // Clear advance_requested in case another system (e.g. FIRE) set it during
    // the ghost update iteration that runs before this check.
    if remaining == 0 {
        scheduler_manager.advance_requested = false;
        scheduler_manager.index += 1;
        scheduler_manager.state = SchedulerState::Setup;
        if scheduler_manager.index >= run_config.num_stages() {
            scheduler_manager.state = SchedulerState::End;
        }
        return;
    }

    run_state.cycle_count[index] += 1;
    run_state.total_cycle += 1;

    let steps_done = run_state.cycle_count[index] == run_state.cycle_remaining[index];
    let advance = scheduler_manager.advance_requested;

    if steps_done || advance {
        scheduler_manager.advance_requested = false;
        scheduler_manager.index += 1;
        scheduler_manager.state = SchedulerState::Setup;
        if scheduler_manager.index >= run_config.num_stages() {
            scheduler_manager.state = SchedulerState::End;
        }
    }
}

/// Merged config table: global defaults deep-merged with current stage overrides.
pub struct StageOverrides {
    pub table: toml::Table,
}

impl StageOverrides {
    /// Deserialize a section from the merged config, falling back to `T::default()`.
    pub fn section<T: serde::de::DeserializeOwned + Default>(&self, key: &str) -> T {
        self.table
            .get(key)
            .and_then(|v| v.clone().try_into().ok())
            .unwrap_or_default()
    }
}

/// Deep-merge `overrides` into `base`, recursively merging sub-tables.
pub fn deep_merge(base: &mut toml::Table, overrides: &toml::Table) {
    for (key, value) in overrides {
        match (base.get_mut(key), value) {
            (Some(toml::Value::Table(base_table)), toml::Value::Table(override_table)) => {
                deep_merge(base_table, override_table);
            }
            _ => {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}

/// Setup system: copies stage name from RunConfig into SchedulerManager, and
/// applies stage config overrides by merging with global config.
/// Config sections that are only read during the first stage.
/// If a later `[[run]]` block overrides these, the override is silently ignored,
/// so we warn the user.
const FIRST_STAGE_ONLY_CONFIGS: &[(&str, &str)] = &[
    ("lattice", "lattice insertion (fcc_insert)"),
    ("lj", "LJ tail corrections (setup_lj_tails)"),
    ("bond", "auto-bonding (auto_bond_touching)"),
    ("comm", "communicator setup (comm_read_input)"),
    ("restart", "restart file reading (read_restart)"),
];

pub fn set_stage_name(
    run_config: Res<RunConfig>,
    config: Res<Config>,
    comm: Res<CommResource>,
    mut scheduler_manager: ResMut<SchedulerManager>,
    mut stage_overrides: ResMut<StageOverrides>,
) {
    let index = scheduler_manager.index;
    if index >= run_config.num_stages() {
        return;
    }
    let stage = run_config.current_stage(index);
    scheduler_manager.stage_name = stage.name.clone();

    // Warn if later stages override config sections that are only read in the first stage
    if index > 0 && comm.rank() == 0 {
        for &(section, description) in FIRST_STAGE_ONLY_CONFIGS {
            if stage.overrides.contains_key(section) {
                let stage_label = stage.name.as_deref().unwrap_or("unnamed");
                eprintln!(
                    "WARNING: Stage {} [{}] overrides [{}], but {} only runs in the first stage. \
                     This override will be ignored.",
                    index, stage_label, section, description
                );
            }
        }
    }

    // Build merged config: global defaults + stage overrides
    let mut merged = config.table.clone();
    // Remove "run" from merged since it's not a config section
    merged.remove("run");
    deep_merge(&mut merged, &stage.overrides);
    stage_overrides.table = merged;
}

/// Setup system: validates that stage names match StageNames (from StageAdvancePlugin).
pub fn validate_stages(
    run_config: Res<RunConfig>,
    stage_names: Res<StageNames>,
    scheduler_manager: Res<SchedulerManager>,
) {
    // Only validate on first stage
    if scheduler_manager.index != 0 {
        return;
    }

    let expected = stage_names.0;
    let actual: Vec<Option<&str>> = run_config
        .stages
        .iter()
        .map(|s| s.name.as_deref())
        .collect();

    // Check count matches
    if run_config.stages.len() != expected.len() {
        panic!(
            "Stage count mismatch: {} [[run]] stages in TOML, but StageEnum has {} variants.\n\
             Expected stage names: {:?}\n\
             TOML stage names: {:?}",
            run_config.stages.len(),
            expected.len(),
            expected,
            actual,
        );
    }

    // Check each stage name matches
    for (i, (expected_name, stage)) in expected.iter().zip(run_config.stages.iter()).enumerate() {
        match &stage.name {
            Some(name) if name != expected_name => {
                panic!(
                    "Stage {} name mismatch: TOML has \"{}\", but StageEnum expects \"{}\"",
                    i, name, expected_name,
                );
            }
            None => {
                panic!(
                    "Stage {} is missing a name in TOML. Expected name: \"{}\"\n\
                     All [[run]] stages must have a `name` when using StageAdvancePlugin.",
                    i, expected_name,
                );
            }
            _ => {} // matches
        }
    }
}

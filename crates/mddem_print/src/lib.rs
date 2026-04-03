//! Output systems for MDDEM simulations: thermo printing, CSV/binary dump files,
//! restart file serialization, and VTP (ParaView) visualization output.
//!
//! # Overview
//!
//! This crate provides four output subsystems, each controlled by its own TOML
//! configuration section:
//!
//! | System  | TOML section  | Description                                    |
//! |---------|---------------|------------------------------------------------|
//! | Thermo  | `[thermo]`    | Periodic console output of simulation metrics  |
//! | Dump    | `[dump]`      | Per-atom CSV or binary snapshots                |
//! | Restart | `[restart]`   | Checkpoint files for resuming simulations       |
//! | VTP     | `[vtp]`       | ParaView-compatible `.vtp` visualization files  |
//!
//! All systems are registered automatically when [`PrintPlugin`] is added to the app.
//!
//! # TOML Configuration
//!
//! ```toml
//! [thermo]
//! # Columns to display (optional — defaults to step, atoms, ke, neighbors, walltime, stepps).
//! # Use "compute/group" syntax for group-filtered values, e.g. "ke/mobile".
//! # Built-in columns: step, atoms, ke, temp, neighbors, walltime, stepps.
//! # Any name pushed via Thermo::set() is also available.
//! columns = ["step", "atoms", "ke", "temp", "walltime", "stepps"]
//!
//! [dump]
//! # Write dump every N steps (0 = disabled, default: 0)
//! interval = 1000
//! # Output format: "text" (CSV) or "binary" (little-endian f64/u32)
//! format = "text"
//!
//! [restart]
//! # Write restart every N steps (0 = disabled, default: 0)
//! interval = 5000
//! # File format: "bincode" (compact binary, default) or "json" (human-readable)
//! format = "bincode"
//! # Read the latest restart file at startup (default: false)
//! read = false
//!
//! [vtp]
//! # Write VTP (ParaView) output every N steps (0 = disabled, default: 0)
//! interval = 500
//! ```
//!
//! # Extending Dump / VTP Output
//!
//! Plugins can register additional per-atom columns via [`DumpRegistry`]:
//!
//! ```rust,ignore
//! let dump_reg = app.get_resource_mut::<DumpRegistry>().unwrap();
//! dump_reg.register_scalar("pressure", |atoms, registry| {
//!     // Return Vec<f64> of length atoms.nlocal
//!     vec![0.0; atoms.nlocal as usize]
//! });
//! ```

use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufWriter, Write},
    time::Instant,
};

use sim_app::prelude::*;
use sim_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

use mddem_core::{compute_ke, Atom, AtomDataRegistry, CommResource, Config, GroupRegistry, Input, RunConfig, RunState, ParticleSimScheduleSet, ScheduleSetupSet, VirialStress};
use mddem_neighbor::Neighbor;

// ── Thermo config ───────────────────────────────────────────────────────────

/// TOML `[thermo]` — configures which columns appear in thermo console output.
///
/// # TOML Fields
///
/// | Field     | Type             | Default                                             | Description                        |
/// |-----------|------------------|------------------------------------------------------|------------------------------------|
/// | `columns` | `[String]` (opt) | `["step","atoms","ke","neighbors","walltime","stepps"]` | Column names to display         |
///
/// Column names can use `"compute/group"` syntax (e.g. `"ke/mobile"`) to filter
/// by a named atom group. Built-in compute names: `step`, `atoms`, `ke`, `temp`,
/// `neighbors`, `walltime`, `stepps`. Any value pushed via [`Thermo::set`] is also
/// available as a column.
#[derive(Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct ThermoConfig {
    /// Column names to display. If `None`, uses the default set.
    #[serde(default)]
    pub columns: Option<Vec<String>>,
}

// ── Thermo column ───────────────────────────────────────────────────────────

/// A parsed thermo column specification, produced from a raw string like `"ke/mobile"`.
pub struct ThermoColumn {
    /// The original column string from config (e.g. `"ke/mobile"`).
    pub raw: String,
    /// The compute name portion (e.g. `"ke"`).
    pub compute_name: String,
    /// Optional group name filter (e.g. `Some("mobile")`).
    pub group_name: Option<String>,
    /// Formatted header string for console display (e.g. `"Ke/mobile"`).
    pub header: String,
    /// Column display width in characters (minimum 12).
    pub width: usize,
}

/// Parse a raw column spec string (e.g. `"ke/mobile"`) into a [`ThermoColumn`].
fn parse_thermo_column(raw: &str) -> ThermoColumn {
    let parts: Vec<&str> = raw.splitn(2, '/').collect();
    let compute_name = parts[0].to_string();
    let group_name = parts.get(1).map(|s| s.to_string());

    let header = if let Some(ref g) = group_name {
        format!("{}/{}", capitalize(&compute_name), g)
    } else {
        capitalize(&compute_name)
    };

    let width = header.len().max(12);

    ThermoColumn {
        raw: raw.to_string(),
        compute_name,
        group_name,
        header,
        width,
    }
}

/// Capitalize the first character of a string.
fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// Returns the default thermo column names when no `[thermo] columns` is specified.
fn default_columns() -> Vec<String> {
    vec![
        "step".into(),
        "atoms".into(),
        "ke".into(),
        "neighbors".into(),
        "walltime".into(),
        "stepps".into(),
    ]
}

// ── Thermo ──────────────────────────────────────────────────────────────────

/// Runtime state for thermo console output.
///
/// Tracks the print interval, wall-clock timing for steps-per-second calculation,
/// parsed column specifications, and user-pushed values from other plugins.
///
/// Other plugins can push custom values via [`Thermo::set`], which become available
/// as thermo columns if listed in the `[thermo] columns` config.
pub struct Thermo {
    /// Print thermo output every N steps.
    pub interval: usize,
    /// Wall-clock timestamp of the last thermo print (for steps/sec calculation).
    pub start_time: Instant,
    /// The step number at which thermo was last printed.
    pub last_printed_step: usize,
    /// Parsed column specifications for output formatting.
    pub columns: Vec<ThermoColumn>,
    /// User-pushed named values (e.g. `"pe"` → `42.0`), available as thermo columns.
    pub values: HashMap<String, f64>,
}

impl Default for Thermo {
    fn default() -> Self {
        Self::new()
    }
}

impl Thermo {
    /// Create a new `Thermo` with default interval of 100 steps.
    pub fn new() -> Self {
        Thermo {
            interval: 100,
            start_time: Instant::now(),
            last_printed_step: 0,
            columns: Vec::new(),
            values: HashMap::new(),
        }
    }

    /// Push a named value into the thermo value map.
    ///
    /// The value becomes available as a thermo column if its name is listed in
    /// `[thermo] columns`. Values are overwritten on each call, so plugins should
    /// call this every thermo interval to keep values current.
    pub fn set(&mut self, name: &str, value: f64) {
        self.values.insert(name.to_string(), value);
    }
}

// ── Dump config ─────────────────────────────────────────────────────────────

/// Default dump format: CSV text.
fn default_dump_format() -> String {
    "text".to_string()
}

/// TOML `[dump]` — per-atom dump file output settings.
///
/// # TOML Fields
///
/// | Field      | Type   | Default  | Description                              |
/// |------------|--------|----------|------------------------------------------|
/// | `interval` | `usize`| `0`      | Write dump every N steps (0 = disabled)  |
/// | `format`   | `String`| `"text"`| `"text"` (CSV) or `"binary"` (little-endian) |
///
/// # Output Files
///
/// - **Text**: `dump/dump_{step}_rank{rank}.csv` — CSV with header row
/// - **Binary**: `dump/dump_{step}_rank{rank}.bin` — `u32` count, then per-atom
///   fields as little-endian `u32`/`f64`
#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DumpConfig {
    /// Write dump every N steps (0 = disabled).
    #[serde(default)]
    pub interval: usize,
    /// Output format: `"text"` (CSV) or `"binary"` (little-endian).
    #[serde(default = "default_dump_format")]
    pub format: String,
}

impl Default for DumpConfig {
    fn default() -> Self {
        DumpConfig {
            interval: 0,
            format: "text".to_string(),
        }
    }
}

// ── Restart config ──────────────────────────────────────────────────────────

/// Default restart format: bincode.
fn default_restart_format() -> String {
    "bincode".to_string()
}

/// TOML `[restart]` — restart (checkpoint) file write/read settings.
///
/// # TOML Fields
///
/// | Field      | Type    | Default      | Description                                  |
/// |------------|---------|--------------|----------------------------------------------|
/// | `interval` | `usize` | `0`         | Write restart every N steps (0 = disabled)   |
/// | `format`   | `String`| `"bincode"` | `"bincode"` (compact) or `"json"` (readable) |
/// | `read`     | `bool`  | `false`     | Read latest restart file at startup          |
///
/// When `read = true`, the system scans the restart directory for the highest-numbered
/// restart file matching this rank and format, then restores atom state from it.
#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct RestartConfig {
    /// Write restart every N steps (0 = disabled).
    #[serde(default)]
    pub interval: usize,
    /// File format: `"bincode"` (compact binary) or `"json"` (human-readable).
    #[serde(default = "default_restart_format")]
    pub format: String,
    /// Whether to read the latest restart file at startup.
    #[serde(default)]
    pub read: bool,
}

impl Default for RestartConfig {
    fn default() -> Self {
        RestartConfig {
            interval: 0,
            format: "bincode".to_string(),
            read: false,
        }
    }
}

// ── RestartData ─────────────────────────────────────────────────────────────

/// Serializable snapshot of all atom state for restart files.
///
/// Positions, velocities, and forces are stored as separate x/y/z vectors for
/// serialization compatibility. Legacy rotational fields (`omega_*`, `torque_*`,
/// etc.) are kept for backwards compatibility with old restart files but are no
/// longer written — rotational data now lives in `atom_data_buffers` via `AtomData`.
#[derive(Serialize, Deserialize)]
struct RestartData {
    natoms: u64,
    total_cycle: usize,
    dt: f64,
    tag: Vec<u32>,
    atom_type: Vec<u32>,
    pos_x: Vec<f64>,
    pos_y: Vec<f64>,
    pos_z: Vec<f64>,
    vel_x: Vec<f64>,
    vel_y: Vec<f64>,
    vel_z: Vec<f64>,
    force_x: Vec<f64>,
    force_y: Vec<f64>,
    force_z: Vec<f64>,
    mass: Vec<f64>,
    cutoff_radius: Vec<f64>,
    atom_data_buffers: Vec<Vec<f64>>,
    // Legacy fields for backwards compatibility with old restart files
    #[serde(default)]
    omega_x: Vec<f64>,
    #[serde(default)]
    omega_y: Vec<f64>,
    #[serde(default)]
    omega_z: Vec<f64>,
    #[serde(default)]
    torque_x: Vec<f64>,
    #[serde(default)]
    torque_y: Vec<f64>,
    #[serde(default)]
    torque_z: Vec<f64>,
    #[serde(default)]
    ang_mom_x: Vec<f64>,
    #[serde(default)]
    ang_mom_y: Vec<f64>,
    #[serde(default)]
    ang_mom_z: Vec<f64>,
    #[serde(default)]
    quaternion: Vec<[f64; 4]>,
}

impl RestartData {
    /// Build a `RestartData` snapshot from the current atom state.
    ///
    /// Only local atoms (indices `0..nlocal`) are included; ghost atoms are excluded.
    fn from_atoms(atoms: &Atom, registry: &AtomDataRegistry, step: usize) -> Self {
        let nlocal = atoms.nlocal as usize;
        RestartData {
            natoms: atoms.natoms,
            total_cycle: step,
            dt: atoms.dt,
            tag: atoms.tag[..nlocal].to_vec(),
            atom_type: atoms.atom_type[..nlocal].to_vec(),
            pos_x: atoms.pos[..nlocal].iter().map(|p| p[0]).collect(),
            pos_y: atoms.pos[..nlocal].iter().map(|p| p[1]).collect(),
            pos_z: atoms.pos[..nlocal].iter().map(|p| p[2]).collect(),
            vel_x: atoms.vel[..nlocal].iter().map(|v| v[0]).collect(),
            vel_y: atoms.vel[..nlocal].iter().map(|v| v[1]).collect(),
            vel_z: atoms.vel[..nlocal].iter().map(|v| v[2]).collect(),
            force_x: atoms.force[..nlocal].iter().map(|v| v[0]).collect(),
            force_y: atoms.force[..nlocal].iter().map(|v| v[1]).collect(),
            force_z: atoms.force[..nlocal].iter().map(|v| v[2]).collect(),
            mass: atoms.mass[..nlocal].to_vec(),
            cutoff_radius: atoms.cutoff_radius[..nlocal].to_vec(),
            atom_data_buffers: registry.pack_all_for_restart(nlocal),
            // Legacy fields left empty — rotational data now in atom_data_buffers via DemAtom
            omega_x: Vec::new(),
            omega_y: Vec::new(),
            omega_z: Vec::new(),
            torque_x: Vec::new(),
            torque_y: Vec::new(),
            torque_z: Vec::new(),
            ang_mom_x: Vec::new(),
            ang_mom_y: Vec::new(),
            ang_mom_z: Vec::new(),
            quaternion: Vec::new(),
        }
    }
}

// ── DumpRegistry ────────────────────────────────────────────────────────

/// Registry of user-defined per-atom data callbacks for dump and VTP output.
///
/// Plugins register callbacks during their `build()` phase. These callbacks are
/// only invoked on steps when dump/VTP output is actually written — zero overhead
/// on non-output steps.
///
/// # Example
///
/// ```rust,ignore
/// // In a plugin's build():
/// let dump_reg = app.get_resource_mut::<DumpRegistry>().unwrap();
/// dump_reg.register_scalar("pressure", |atoms, registry| {
///     let dem = registry.expect::<DemAtom>("pressure");
///     (0..atoms.nlocal as usize).map(|i| /* ... */ 0.0).collect()
/// });
/// ```
pub struct DumpRegistry {
    scalar_fns: Vec<(
        String,
        Box<dyn Fn(&Atom, &AtomDataRegistry) -> Vec<f64> + Send + Sync>,
    )>,
    vector_fns: Vec<(
        String,
        Box<dyn Fn(&Atom, &AtomDataRegistry) -> Vec<[f64; 3]> + Send + Sync>,
    )>,
}

impl Default for DumpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DumpRegistry {
    /// Create an empty `DumpRegistry` with no registered callbacks.
    pub fn new() -> Self {
        DumpRegistry {
            scalar_fns: Vec::new(),
            vector_fns: Vec::new(),
        }
    }

    /// Register a per-atom scalar column for dump/VTP output.
    ///
    /// The callback receives the current [`Atom`] and [`AtomDataRegistry`] and should
    /// return a `Vec<f64>` of length `atoms.nlocal`. The column appears in CSV dumps
    /// and as a VTP `Float32` data array.
    pub fn register_scalar(
        &mut self,
        name: impl Into<String>,
        f: impl Fn(&Atom, &AtomDataRegistry) -> Vec<f64> + Send + Sync + 'static,
    ) {
        self.scalar_fns.push((name.into(), Box::new(f)));
    }

    /// Register a per-atom 3-component vector column for dump/VTP output.
    ///
    /// The callback should return a `Vec<[f64; 3]>` of length `atoms.nlocal`.
    /// In CSV dumps, the vector is split into `{name}_x`, `{name}_y`, `{name}_z`
    /// columns. In VTP output, it appears as a 3-component `Float32` data array.
    pub fn register_vector(
        &mut self,
        name: impl Into<String>,
        f: impl Fn(&Atom, &AtomDataRegistry) -> Vec<[f64; 3]> + Send + Sync + 'static,
    ) {
        self.vector_fns.push((name.into(), Box::new(f)));
    }

    /// Returns `true` if any scalar or vector callbacks are registered.
    pub fn has_callbacks(&self) -> bool {
        !self.scalar_fns.is_empty() || !self.vector_fns.is_empty()
    }
}

// ── VTP config ──────────────────────────────────────────────────────────────

/// TOML `[vtp]` — ParaView `.vtp` visualization output settings.
///
/// # TOML Fields
///
/// | Field      | Type    | Default | Description                              |
/// |------------|---------|---------|------------------------------------------|
/// | `interval` | `usize` | `0`    | Write VTP every N steps (0 = disabled)   |
///
/// VTP files are written to `{output_dir}/vtp/{step}CYCLE_{rank}RANK.vtp` and
/// include particle positions, radii, velocity magnitudes, ghost flags, and any
/// fields registered via [`DumpRegistry`].
#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct VtpConfig {
    /// Write VTP every N steps (0 = disabled).
    #[serde(default)]
    pub interval: usize,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Main output plugin — registers thermo, dump, restart, and VTP systems.
///
/// Add this plugin to the app to enable all output subsystems. Each subsystem
/// is independently configured via its TOML section and only produces output
/// when its interval is non-zero.
pub struct PrintPlugin;

impl Plugin for PrintPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[dump]
# Dump output interval (0 = disabled)
interval = 0
# Dump format: "text" (CSV) or "binary"
format = "text"

[restart]
# Restart file write interval (0 = disabled)
interval = 0
# Restart format: "bincode" or "json"
format = "bincode"
# Whether to read restart files at startup
read = false

[vtp]
# VTP (ParaView) output interval (0 = disabled)
interval = 0"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<DumpConfig>(app, "dump");
        Config::load::<RestartConfig>(app, "restart");
        Config::load::<VtpConfig>(app, "vtp");
        Config::load::<ThermoConfig>(app, "thermo");

        app.add_resource(DumpRegistry::new());
        app.add_resource(Thermo::new())
            .add_setup_system(setup_thermo, ScheduleSetupSet::PostSetup)
            .add_setup_system(read_restart.run_if(first_stage_only()), ScheduleSetupSet::PostSetup)
            .add_update_system(output_virial_to_thermo, ParticleSimScheduleSet::PostForce)
            .add_update_system(print_vtp, ParticleSimScheduleSet::PostFinalIntegration)
            .add_update_system(print_thermo, ParticleSimScheduleSet::PostFinalIntegration)
            .add_update_system(dump_atoms, ParticleSimScheduleSet::PostFinalIntegration)
            .add_update_system(write_restart, ParticleSimScheduleSet::PostFinalIntegration)
            .add_update_system(check_stage_end_save.before("update_cycle"), ParticleSimScheduleSet::PostFinalIntegration);
    }
}

// ── Helper: restart directory ───────────────────────────────────────────────

/// Compute the restart base directory from the output directory setting.
fn restart_base_dir(input: &Input) -> String {
    match input.output_dir.as_deref() {
        Some(dir) => format!("{}/restart", dir),
        None => "restart".to_string(),
    }
}

// ── Thermo systems ──────────────────────────────────────────────────────────

/// Setup system for thermo output: parses column specs and prints the header.
///
/// Runs at the start of each stage to update the print interval and reset the
/// wall-clock timer. Column specifications are parsed only once (on the first stage).
pub fn setup_thermo(
    config: Res<RunConfig>,
    thermo_config: Res<ThermoConfig>,
    scheduler_manager: Res<SchedulerManager>,
    comm: Res<CommResource>,
    run_state: Res<RunState>,
    mut thermo: ResMut<Thermo>,
    mut virial: Option<ResMut<VirialStress>>,
) {
    let index = scheduler_manager.index;
    if index >= config.num_stages() {
        return;
    }
    thermo.interval = config.current_stage(index).thermo;
    if let Some(ref mut v) = virial {
        v.set_interval(thermo.interval);
    }
    thermo.start_time = Instant::now();
    thermo.last_printed_step = run_state.total_cycle;

    // Parse column specifications (only on first stage, or if columns empty).
    if thermo.columns.is_empty() {
        let col_names = thermo_config
            .columns
            .clone()
            .unwrap_or_else(default_columns);
        thermo.columns = col_names.iter().map(|s| parse_thermo_column(s)).collect();
    }

    if comm.rank() == 0 {
        println!();
        let header: String = thermo
            .columns
            .iter()
            .map(|c| format!("{:<width$}", c.header, width = c.width))
            .collect::<Vec<_>>()
            .join(" ");
        let total_width: usize = thermo.columns.iter().map(|c| c.width).sum::<usize>()
            + thermo.columns.len().saturating_sub(1);
        println!("{}", header);
        println!("{}", "-".repeat(total_width));
    }
}

/// Print thermo output to console at the configured interval.
///
/// All MPI ranks participate in allreduce operations (for KE, atom counts, etc.),
/// but only rank 0 prints the formatted output line.
#[allow(clippy::too_many_arguments)]
pub fn print_thermo(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    neighbor: Res<Neighbor>,
    groups: Res<GroupRegistry>,
    mut thermo: ResMut<Thermo>,
) {
    let step = run_state.total_cycle;
    if !step.is_multiple_of(thermo.interval) {
        return;
    }

    // Pre-compute values that need allreduce (all ranks must participate).
    // We compute these eagerly so all ranks call allreduce together.
    let local_ke_all = compute_ke(&atoms, None);
    let global_ke_all = comm.all_reduce_sum_f64(local_ke_all);
    let global_atoms_all = atoms.natoms as f64;
    let local_neighbors = neighbor.neighbor_indices.len() as f64;
    let global_neighbors = comm.all_reduce_sum_f64(local_neighbors);

    // Pre-compute group-filtered KE and atom counts for any group columns.
    // Each group that appears needs its own allreduce.
    let mut group_ke: HashMap<String, f64> = HashMap::new();
    let mut group_count: HashMap<String, f64> = HashMap::new();
    for col in thermo.columns.iter() {
        if let Some(ref gname) = col.group_name {
            if group_ke.contains_key(gname) {
                continue;
            }
            if let Some(group) = groups.get(gname) {
                let local_ke = compute_ke(&atoms, Some(&group.mask));
                let local_count = group.count as f64;
                group_ke.insert(gname.clone(), comm.all_reduce_sum_f64(local_ke));
                group_count.insert(gname.clone(), comm.all_reduce_sum_f64(local_count));
            }
        }
    }

    if comm.rank() == 0 {
        let elapsed = thermo.start_time.elapsed().as_secs_f64();
        let steps_since = (step - thermo.last_printed_step) as f64;
        let steps_per_sec = if elapsed > 1e-9 {
            steps_since / elapsed
        } else {
            0.0
        };

        let mut parts: Vec<String> = Vec::new();
        for col in thermo.columns.iter() {
            let val_str = match col.compute_name.as_str() {
                "step" => format!("{:<width$}", step, width = col.width),
                "atoms" => {
                    if let Some(ref gname) = col.group_name {
                        let n = group_count.get(gname).copied().unwrap_or(0.0) as u64;
                        format!("{:<width$}", n, width = col.width)
                    } else {
                        format!("{:<width$}", atoms.natoms, width = col.width)
                    }
                }
                "ke" => {
                    let ke = if let Some(ref gname) = col.group_name {
                        group_ke.get(gname).copied().unwrap_or(0.0)
                    } else {
                        global_ke_all
                    };
                    format!("{:<width$.6e}", ke, width = col.width)
                }
                "temp" => {
                    let (ke, n) = if let Some(ref gname) = col.group_name {
                        (
                            group_ke.get(gname).copied().unwrap_or(0.0),
                            group_count.get(gname).copied().unwrap_or(0.0),
                        )
                    } else {
                        (global_ke_all, global_atoms_all)
                    };
                    let ndof = 3.0 * n - 3.0;
                    let temp = if ndof > 0.0 { 2.0 * ke / ndof } else { 0.0 };
                    format!("{:<width$.6}", temp, width = col.width)
                }
                "neighbors" => {
                    format!("{:<width$}", global_neighbors as usize, width = col.width)
                }
                "walltime" => {
                    format!("{:<width$.4}", elapsed, width = col.width)
                }
                "stepps" => {
                    format!("{:<width$.1}", steps_per_sec, width = col.width)
                }
                other => {
                    // User-pushed value from Thermo::set()
                    if let Some(&v) = thermo.values.get(other) {
                        format!("{:<width$.6e}", v, width = col.width)
                    } else {
                        format!("{:<width$}", "N/A", width = col.width)
                    }
                }
            };
            parts.push(val_str);
        }
        println!("{}", parts.join(" "));
        thermo.start_time = Instant::now();
        thermo.last_printed_step = step;
    }
}

// ── Virial stress output ────────────────────────────────────────────────────

/// MPI-reduce each virial stress component and push to thermo values.
///
/// Publishes `virial_xx`, `virial_yy`, `virial_zz`, `virial_xy`, `virial_xz`,
/// `virial_yz` as thermo columns. Only runs on thermo output steps.
pub fn output_virial_to_thermo(
    virial: Option<Res<VirialStress>>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    mut thermo: ResMut<Thermo>,
) {
    if !run_state.total_cycle.is_multiple_of(thermo.interval) {
        return;
    }
    let virial = match virial {
        Some(v) => v,
        None => return,
    };
    let xx = comm.all_reduce_sum_f64(virial.xx);
    let yy = comm.all_reduce_sum_f64(virial.yy);
    let zz = comm.all_reduce_sum_f64(virial.zz);
    let xy = comm.all_reduce_sum_f64(virial.xy);
    let xz = comm.all_reduce_sum_f64(virial.xz);
    let yz = comm.all_reduce_sum_f64(virial.yz);
    thermo.set("virial_xx", xx);
    thermo.set("virial_yy", yy);
    thermo.set("virial_zz", zz);
    thermo.set("virial_xy", xy);
    thermo.set("virial_xz", xz);
    thermo.set("virial_yz", yz);
}

// ── VTP output ──────────────────────────────────────────────────────────────

/// Write a single VTP `<DataArray>` element with per-point scalar data.
fn write_vtp_data_array(
    file: &mut File,
    vtp_type: &str,
    name: &str,
    n: usize,
    value_fn: impl Fn(usize) -> String,
) -> std::io::Result<()> {
    writeln!(
        file,
        "<DataArray type=\"{}\" Name=\"{}\" format=\"ascii\">",
        vtp_type, name
    )?;
    for i in 0..n {
        writeln!(file, "{}", value_fn(i))?;
    }
    write!(file, "</DataArray>")?;
    Ok(())
}

/// Write ParaView VTP output at the configured interval.
///
/// Each MPI rank writes its own `.vtp` file containing local + ghost atoms.
/// Output includes positions, radii, velocity magnitude, ghost flags, and any
/// fields registered via [`DumpRegistry`].
#[allow(clippy::too_many_arguments)]
pub fn print_vtp(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    vtp_config: Res<VtpConfig>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
    dump_registry: Res<DumpRegistry>,
) {
    let count = run_state.total_cycle;
    let rank = comm.rank();
    let stage = run_config.current_stage(scheduler_manager.index);
    let interval = stage.vtp_interval.unwrap_or(vtp_config.interval);
    if interval == 0 || !count.is_multiple_of(interval) {
        return;
    }
    if let Err(e) = print_vtp_inner(&atoms, &registry, count, rank, &input, &dump_registry) {
        eprintln!("WARNING: VTP write failed at step {}: {}", count, e);
    }
}

/// Inner VTP write logic, separated for error handling via `?`.
fn print_vtp_inner(
    atoms: &Atom,
    registry: &AtomDataRegistry,
    count: usize,
    rank: i32,
    input: &Input,
    dump_reg: &DumpRegistry,
) -> std::io::Result<()> {
    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("{}/vtp", dir),
        None => "vtp".to_string(),
    };
    let filename = format!("{}/{}CYCLE_{}RANK.vtp", base_dir, count, rank);
    fs::create_dir_all(&base_dir)?;
    let mut file = File::create(&filename)?;

    let n = atoms.len();
    let nlocal = atoms.nlocal as usize;

    // XML header and PolyData opening
    write!(&mut file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n<PolyData>\n")?;
    writeln!(&mut file, "<Piece NumberOfPoints=\"{}\">", n)?;

    // Points (positions)
    write!(&mut file, "<Points><DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">")?;
    for i in 0..n {
        writeln!(&mut file, "{} {} {}", atoms.pos[i][0], atoms.pos[i][1], atoms.pos[i][2])?;
    }
    write!(&mut file, "</DataArray>\n</Points>\n")?;

    // Per-point data arrays
    writeln!(&mut file, "<PointData Scalars=\"\" Vectors=\"\">")?;
    write_vtp_data_array(&mut file, "Float32", "Radius", n, |i| format!("{}", atoms.cutoff_radius[i]))?;
    write_vtp_data_array(&mut file, "Float32", "Vel_Mag", n, |i| {
        let vmag = (atoms.vel[i][0].powi(2) + atoms.vel[i][1].powi(2) + atoms.vel[i][2].powi(2)).sqrt();
        format!("{vmag}")
    })?;
    write_vtp_data_array(&mut file, "Int32", "IsGhost", n, |i| {
        if i >= nlocal { "1".to_string() } else { "0".to_string() }
    })?;

    // Registered scalar callbacks
    for (name, f) in &dump_reg.scalar_fns {
        let data = f(atoms, registry);
        write_vtp_data_array(&mut file, "Float32", name, n, |i| {
            if i < data.len() {
                format!("{}", data[i])
            } else {
                "0".to_string()
            }
        })?;
    }

    // Registered vector callbacks
    for (name, f) in &dump_reg.vector_fns {
        let data = f(atoms, registry);
        writeln!(
            &mut file,
            "<DataArray type=\"Float32\" Name=\"{}\" NumberOfComponents=\"3\" format=\"ascii\">",
            name
        )?;
        for i in 0..n {
            if i < data.len() {
                writeln!(&mut file, "{} {} {}", data[i][0], data[i][1], data[i][2])?;
            } else {
                writeln!(&mut file, "0 0 0")?;
            }
        }
        write!(&mut file, "</DataArray>")?;
    }

    write!(&mut file, "</PointData>\n</Piece>\n</PolyData>\n</VTKFile>\n")?;
    Ok(())
}

// ── Dump output ─────────────────────────────────────────────────────────────

/// Write per-atom dump files (CSV or binary) at the configured interval.
///
/// Each MPI rank writes its own file containing only local atoms. The dump
/// includes core fields (tag, type, position, velocity, force, radius) plus
/// any columns registered via [`DumpRegistry`].
#[allow(clippy::too_many_arguments)]
pub fn dump_atoms(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    dump_config: Res<DumpConfig>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
    dump_registry: Res<DumpRegistry>,
) {
    let stage = run_config.current_stage(scheduler_manager.index);
    let interval = stage.dump_interval.unwrap_or(dump_config.interval);
    if interval == 0 {
        return;
    }
    let step = run_state.total_cycle;
    if !step.is_multiple_of(interval) {
        return;
    }

    if let Err(e) = dump_atoms_inner(&atoms, &registry, step, comm.rank(), &input, &dump_config, &dump_registry) {
        eprintln!("WARNING: Dump write failed at step {}: {}", step, e);
    }
}

/// Inner dump write logic, shared between periodic dumps and stage-end saves.
pub(crate) fn dump_atoms_inner(
    atoms: &Atom,
    registry: &AtomDataRegistry,
    step: usize,
    rank: i32,
    input: &Input,
    dump_config: &DumpConfig,
    dump_reg: &DumpRegistry,
) -> std::io::Result<()> {
    let nlocal = atoms.nlocal as usize;
    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("{}/dump", dir),
        None => "dump".to_string(),
    };
    fs::create_dir_all(&base_dir)?;

    // Evaluate registered callbacks (only when dump is actually written)
    let scalar_data: Vec<(&str, Vec<f64>)> = dump_reg
        .scalar_fns
        .iter()
        .map(|(name, f)| (name.as_str(), f(atoms, registry)))
        .collect();
    let vector_data: Vec<(&str, Vec<[f64; 3]>)> = dump_reg
        .vector_fns
        .iter()
        .map(|(name, f)| (name.as_str(), f(atoms, registry)))
        .collect();

    match dump_config.format.as_str() {
        "binary" => {
            let filename = format!("{}/dump_{}_rank{}.bin", base_dir, step, rank);
            let file = File::create(&filename)?;
            let mut w = BufWriter::new(file);
            w.write_all(&(nlocal as u32).to_le_bytes())?;
            for i in 0..nlocal {
                w.write_all(&atoms.tag[i].to_le_bytes())?;
                w.write_all(&atoms.atom_type[i].to_le_bytes())?;
                w.write_all(&atoms.pos[i][0].to_le_bytes())?;
                w.write_all(&atoms.pos[i][1].to_le_bytes())?;
                w.write_all(&atoms.pos[i][2].to_le_bytes())?;
                w.write_all(&atoms.vel[i][0].to_le_bytes())?;
                w.write_all(&atoms.vel[i][1].to_le_bytes())?;
                w.write_all(&atoms.vel[i][2].to_le_bytes())?;
                w.write_all(&atoms.force[i][0].to_le_bytes())?;
                w.write_all(&atoms.force[i][1].to_le_bytes())?;
                w.write_all(&atoms.force[i][2].to_le_bytes())?;
                w.write_all(&atoms.cutoff_radius[i].to_le_bytes())?;
                for (_, data) in &scalar_data {
                    w.write_all(&data[i].to_le_bytes())?;
                }
                for (_, data) in &vector_data {
                    w.write_all(&data[i][0].to_le_bytes())?;
                    w.write_all(&data[i][1].to_le_bytes())?;
                    w.write_all(&data[i][2].to_le_bytes())?;
                }
            }
        }
        _ => {
            // Default: text/CSV
            let filename = format!("{}/dump_{}_rank{}.csv", base_dir, step, rank);
            let file = File::create(&filename)?;
            let mut w = BufWriter::new(file);

            // Build header
            let mut header = "tag,type,x,y,z,vx,vy,vz,fx,fy,fz,radius".to_string();
            for (name, _) in &scalar_data {
                header.push(',');
                header.push_str(name);
            }
            for (name, _) in &vector_data {
                header.push(',');
                header.push_str(&format!("{name}_x,{name}_y,{name}_z"));
            }
            writeln!(w, "{}", header)?;

            // Write per-atom rows
            for i in 0..nlocal {
                write!(
                    w,
                    "{},{},{},{},{},{},{},{},{},{},{},{}",
                    atoms.tag[i],
                    atoms.atom_type[i],
                    atoms.pos[i][0],
                    atoms.pos[i][1],
                    atoms.pos[i][2],
                    atoms.vel[i][0],
                    atoms.vel[i][1],
                    atoms.vel[i][2],
                    atoms.force[i][0],
                    atoms.force[i][1],
                    atoms.force[i][2],
                    atoms.cutoff_radius[i],
                )?;
                for (_, data) in &scalar_data {
                    write!(w, ",{}", data[i])?;
                }
                for (_, data) in &vector_data {
                    write!(w, ",{},{},{}", data[i][0], data[i][1], data[i][2])?;
                }
                writeln!(w)?;
            }
        }
    }
    Ok(())
}

// ── Restart write ───────────────────────────────────────────────────────────

/// Write restart (checkpoint) files at the configured interval.
///
/// Each MPI rank writes its own restart file containing only its local atoms.
/// The file format is determined by `[restart] format` (bincode or JSON).
#[allow(clippy::too_many_arguments)]
pub fn write_restart(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    restart_config: Res<RestartConfig>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
) {
    let stage = run_config.current_stage(scheduler_manager.index);
    let interval = stage.restart_interval.unwrap_or(restart_config.interval);
    if interval == 0 {
        return;
    }
    let step = run_state.total_cycle;
    if !step.is_multiple_of(interval) {
        return;
    }

    let rank = comm.rank();
    let base_dir = restart_base_dir(&input);
    fs::create_dir_all(&base_dir).ok();

    let data = RestartData::from_atoms(&atoms, &registry, step);

    if let Err(e) = write_restart_inner(&data, &base_dir, step, rank, &restart_config) {
        eprintln!("WARNING: Restart write failed at step {}: {}", step, e);
    }
}

/// Serialize restart data to disk in the configured format (bincode or JSON).
pub(crate) fn write_restart_inner(
    data: &RestartData,
    base_dir: &str,
    step: usize,
    rank: i32,
    restart_config: &RestartConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    match restart_config.format.as_str() {
        "json" => {
            let filename = format!("{}/restart_{}_rank{}.json", base_dir, step, rank);
            let file = File::create(&filename)?;
            serde_json::to_writer(BufWriter::new(file), data)?;
        }
        _ => {
            let filename = format!("{}/restart_{}_rank{}.bin", base_dir, step, rank);
            let file = File::create(&filename)?;
            bincode::serialize_into(BufWriter::new(file), data)?;
        }
    }
    Ok(())
}

// ── Stage-end save ──────────────────────────────────────────────────────────

/// Write dump + restart files when a stage with `save_at_end = true` finishes.
///
/// Runs before `update_cycle` so the stage index is still valid. This ensures
/// that the final state of each stage is captured even if the regular dump/restart
/// intervals don't align with the stage boundary.
#[allow(clippy::too_many_arguments)]
pub fn check_stage_end_save(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    dump_config: Res<DumpConfig>,
    restart_config: Res<RestartConfig>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
    dump_registry: Res<DumpRegistry>,
) {
    let index = scheduler_manager.index;
    if index >= run_config.num_stages() {
        return;
    }
    let stage = run_config.current_stage(index);
    if !stage.save_at_end {
        return;
    }

    let remaining = run_state.cycle_remaining[index];
    // Don't save for skipped stages (remaining == 0)
    if remaining == 0 {
        return;
    }

    // check_stage_end_save runs .before("update_cycle"), so the cycle counter
    // hasn't been incremented for the current step yet.  After update_cycle runs,
    // count will be count+1, which equals remaining on the final step.
    let count = run_state.cycle_count[index];
    let is_last_step = count + 1 == remaining;
    let is_advancing = scheduler_manager.advance_requested;

    if !is_last_step && !is_advancing {
        return;
    }

    let step = run_state.total_cycle;
    let rank = comm.rank();
    let stage_label = stage.name.as_deref().unwrap_or("(unnamed)");

    if rank == 0 {
        println!("Stage {} [{}] finished — saving dump + restart at step {}", index, stage_label, step);
    }

    // Write dump
    if let Err(e) = dump_atoms_inner(&atoms, &registry, step, rank, &input, &dump_config, &dump_registry) {
        eprintln!("WARNING: Stage-end dump write failed at step {}: {}", step, e);
    }

    // Write restart
    let base_dir = restart_base_dir(&input);
    fs::create_dir_all(&base_dir).ok();

    let data = RestartData::from_atoms(&atoms, &registry, step);

    if let Err(e) = write_restart_inner(&data, &base_dir, step, rank, &restart_config) {
        eprintln!("WARNING: Stage-end restart write failed at step {}: {}", step, e);
    }
}

// ── Restart read ────────────────────────────────────────────────────────────

/// Read the latest restart file at startup and restore atom state.
///
/// Scans the restart directory for files matching the current rank and format,
/// selects the one with the highest step number, and deserializes it to restore
/// all atom data (positions, velocities, forces, mass, radius, and any `AtomData`
/// extensions stored in `atom_data_buffers`).
///
/// Only runs when `[restart] read = true` and only on the first stage.
pub fn read_restart(
    restart_config: Res<RestartConfig>,
    comm: Res<CommResource>,
    input: Res<Input>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    mut run_state: ResMut<RunState>,
) {
    if !restart_config.read {
        return;
    }

    let rank = comm.rank();
    let base_dir = restart_base_dir(&input);

    // Find the latest restart file for this rank
    let ext = match restart_config.format.as_str() {
        "json" => "json",
        _ => "bin",
    };

    let prefix = "restart_";
    let suffix = format!("_rank{}.{}", rank, ext);

    let mut latest_step: Option<usize> = None;
    if let Ok(entries) = fs::read_dir(&base_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(prefix) && name.ends_with(&suffix) {
                let mid = &name[prefix.len()..name.len() - suffix.len()];
                if let Ok(step) = mid.parse::<usize>() {
                    if latest_step.map_or(true, |prev| step > prev) {
                        latest_step = Some(step);
                    }
                }
            }
        }
    }

    let step = match latest_step {
        Some(s) => s,
        None => {
            if rank == 0 {
                println!("Restart: no restart files found in {}", base_dir);
            }
            return;
        }
    };

    let filename = format!("{}/restart_{}_rank{}.{}", base_dir, step, rank, ext);
    if rank == 0 {
        println!("Restart: reading from {}", filename);
    }

    let file = File::open(&filename).unwrap_or_else(|e| {
        eprintln!("ERROR: Failed to open restart file '{}': {}", filename, e);
        std::process::exit(1);
    });
    let data: RestartData = match ext {
        "json" => serde_json::from_reader(std::io::BufReader::new(file)).unwrap_or_else(|e| {
            eprintln!("ERROR: Failed to parse restart JSON '{}': {}", filename, e);
            std::process::exit(1);
        }),
        _ => bincode::deserialize_from(std::io::BufReader::new(file)).unwrap_or_else(|e| {
            eprintln!("ERROR: Failed to deserialize restart bincode '{}': {}", filename, e);
            std::process::exit(1);
        }),
    };

    let n = data.tag.len();

    // Clear existing atoms and repopulate from restart data
    atoms.natoms = data.natoms;
    atoms.nlocal = n as u32;
    atoms.nghost = 0;
    atoms.dt = data.dt;

    atoms.tag = data.tag;
    atoms.atom_type = data.atom_type;
    atoms.origin_index = vec![0; n];
    atoms.is_ghost = vec![false; n];
    atoms.pos = data.pos_x.iter().zip(data.pos_y.iter()).zip(data.pos_z.iter())
        .map(|((&x, &y), &z)| [x, y, z]).collect();
    atoms.vel = data.vel_x.iter().zip(data.vel_y.iter()).zip(data.vel_z.iter())
        .map(|((&x, &y), &z)| [x, y, z]).collect();
    atoms.force = data.force_x.iter().zip(data.force_y.iter()).zip(data.force_z.iter())
        .map(|((&x, &y), &z)| [x, y, z]).collect();
    atoms.mass = data.mass;
    atoms.inv_mass = atoms.mass.iter().map(|&m| 1.0 / m).collect();
    atoms.cutoff_radius = data.cutoff_radius;

    // Restore AtomData (DemAtom, etc.) from generic buffers
    if !data.atom_data_buffers.is_empty() {
        registry.unpack_all_from_restart(&data.atom_data_buffers);
    }

    run_state.total_cycle = data.total_cycle;
    if rank == 0 {
        println!("Restart: loaded {} atoms from step {}", n, data.total_cycle);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_columns_backward_compat() {
        let config = ThermoConfig::default();
        assert!(config.columns.is_none());
        let cols = config.columns.unwrap_or_else(default_columns);
        assert_eq!(cols, vec!["step", "atoms", "ke", "neighbors", "walltime", "stepps"]);
    }

    #[test]
    fn test_column_parsing() {
        let col = parse_thermo_column("temp/mobile");
        assert_eq!(col.compute_name, "temp");
        assert_eq!(col.group_name.as_deref(), Some("mobile"));
        assert_eq!(col.header, "Temp/mobile");

        let col2 = parse_thermo_column("step");
        assert_eq!(col2.compute_name, "step");
        assert!(col2.group_name.is_none());
        assert_eq!(col2.header, "Step");
    }

    #[test]
    fn test_user_value_set_and_read() {
        let mut thermo = Thermo::new();
        assert!(thermo.values.get("pe").is_none());
        thermo.set("pe", 42.0);
        assert_eq!(*thermo.values.get("pe").unwrap(), 42.0);
        thermo.set("pe", 99.0);
        assert_eq!(*thermo.values.get("pe").unwrap(), 99.0);
    }

    #[test]
    fn test_capitalize() {
        assert_eq!(capitalize("step"), "Step");
        assert_eq!(capitalize("ke"), "Ke");
        assert_eq!(capitalize(""), "");
        assert_eq!(capitalize("a"), "A");
    }

    #[test]
    fn test_dump_registry_has_callbacks() {
        let mut reg = DumpRegistry::new();
        assert!(!reg.has_callbacks());
        reg.register_scalar("test", |_atoms, _reg| vec![]);
        assert!(reg.has_callbacks());
    }

    #[test]
    fn test_column_width_minimum() {
        let col = parse_thermo_column("ke");
        // "Ke" is 2 chars, but minimum width is 12
        assert_eq!(col.width, 12);
    }

    #[test]
    fn test_column_width_long_header() {
        let col = parse_thermo_column("virial_xx/long_group_name");
        // "Virial_xx/long_group_name" is 25 chars > 12
        assert!(col.width >= 12);
        assert_eq!(col.width, col.header.len());
    }
}

//! Output systems: LAMMPS-style thermo, CSV/binary dump, restart files, and VTP (ParaView) output.

use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufWriter, Write},
    time::Instant,
};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

use mddem_core::{Atom, AtomDataRegistry, CommResource, Config, GroupRegistry, Input, RunConfig, RunState};
use mddem_neighbor::Neighbor;

// ── Thermo config ───────────────────────────────────────────────────────────

#[derive(Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
/// TOML `[thermo]` — thermo output column configuration.
pub struct ThermoConfig {
    #[serde(default)]
    pub columns: Option<Vec<String>>,
}

// ── Thermo column ───────────────────────────────────────────────────────────

/// Parsed column specification for thermo output.
pub struct ThermoColumn {
    pub raw: String,
    pub compute_name: String,
    pub group_name: Option<String>,
    pub header: String,
    pub width: usize,
}

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

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

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

/// Thermo output state: print interval, wall-clock timing, column specs, and user values.
pub struct Thermo {
    pub interval: usize,
    pub start_time: Instant,
    pub last_printed_step: usize,
    pub columns: Vec<ThermoColumn>,
    pub values: HashMap<String, f64>,
}

impl Default for Thermo {
    fn default() -> Self {
        Self::new()
    }
}

impl Thermo {
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
    /// Available as a column if listed in `[thermo] columns`.
    pub fn set(&mut self, name: &str, value: f64) {
        self.values.insert(name.to_string(), value);
    }
}

// ── Dump config ─────────────────────────────────────────────────────────────

fn default_dump_format() -> String {
    "text".to_string()
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[dump]` — atom dump file output settings.
pub struct DumpConfig {
    /// Write dump every N steps (0 = disabled).
    #[serde(default)]
    pub interval: usize,
    /// Output format: `"text"` (CSV) or `"binary"`.
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

fn default_restart_format() -> String {
    "bincode".to_string()
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[restart]` — restart file write/read settings.
pub struct RestartConfig {
    /// Write restart every N steps (0 = disabled).
    #[serde(default)]
    pub interval: usize,
    /// File format: `"bincode"` or `"json"`.
    #[serde(default = "default_restart_format")]
    pub format: String,
    /// Whether to read restart files at startup.
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
    omega_x: Vec<f64>,
    omega_y: Vec<f64>,
    omega_z: Vec<f64>,
    force_x: Vec<f64>,
    force_y: Vec<f64>,
    force_z: Vec<f64>,
    torque_x: Vec<f64>,
    torque_y: Vec<f64>,
    torque_z: Vec<f64>,
    ang_mom_x: Vec<f64>,
    ang_mom_y: Vec<f64>,
    ang_mom_z: Vec<f64>,
    quaternion: Vec<[f64; 4]>,
    mass: Vec<f64>,
    skin: Vec<f64>,
    atom_data_buffers: Vec<Vec<f64>>,
}

// ── VTP config ──────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
/// TOML `[vtp]` — ParaView VTP output settings.
pub struct VtpConfig {
    /// Write VTP every N steps (0 = disabled).
    #[serde(default)]
    pub interval: usize,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Registers thermo, dump, restart, and VTP output systems.
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

        app.add_resource(Thermo::new())
            .add_setup_system(setup_thermo, ScheduleSetupSet::PostSetup)
            .add_setup_system(read_restart, ScheduleSetupSet::PostSetup)
            .add_update_system(print_vtp, ScheduleSet::PostFinalIntegration)
            .add_update_system(print_thermo, ScheduleSet::PostFinalIntegration)
            .add_update_system(dump_atoms, ScheduleSet::PostFinalIntegration)
            .add_update_system(write_restart, ScheduleSet::PostFinalIntegration);
    }
}

// ── Thermo systems ──────────────────────────────────────────────────────────

pub fn setup_thermo(
    config: Res<RunConfig>,
    thermo_config: Res<ThermoConfig>,
    scheduler_manager: Res<SchedulerManager>,
    comm: Res<CommResource>,
    run_state: Res<RunState>,
    mut thermo: ResMut<Thermo>,
) {
    let index = scheduler_manager.index;
    if index >= config.num_stages() {
        return;
    }
    thermo.interval = config.current_stage(index).thermo;
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
    let nlocal = atoms.nlocal as usize;

    // Pre-compute values that need allreduce (all ranks must participate).
    // We compute these eagerly so all ranks call allreduce together.
    let local_ke_all: f64 = (0..nlocal)
        .map(|i| {
            0.5 * atoms.mass[i]
                * (atoms.vel_x[i] * atoms.vel_x[i]
                    + atoms.vel_y[i] * atoms.vel_y[i]
                    + atoms.vel_z[i] * atoms.vel_z[i])
        })
        .sum();
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
                let local_ke: f64 = (0..nlocal)
                    .filter(|&i| group.mask[i])
                    .map(|i| {
                        0.5 * atoms.mass[i]
                            * (atoms.vel_x[i] * atoms.vel_x[i]
                                + atoms.vel_y[i] * atoms.vel_y[i]
                                + atoms.vel_z[i] * atoms.vel_z[i])
                    })
                    .sum();
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
                    // User-pushed value
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

// ── VTP output ──────────────────────────────────────────────────────────────

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

pub fn print_vtp(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    vtp_config: Res<VtpConfig>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
) {
    let count = run_state.total_cycle;
    let rank = comm.rank();
    let stage = run_config.current_stage(scheduler_manager.index);
    let interval = stage.vtp_interval.unwrap_or(vtp_config.interval);
    if interval == 0 || !count.is_multiple_of(interval) {
        return;
    }
    if let Err(e) = print_vtp_inner(&atoms, count, rank, &input) {
        eprintln!("WARNING: VTP write failed at step {}: {}", count, e);
    }
}

fn print_vtp_inner(
    atoms: &Atom,
    count: usize,
    rank: i32,
    input: &Input,
) -> std::io::Result<()> {
    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("./{}/vtp", dir),
        None => "./vtp".to_string(),
    };
    let filename = format!("{}/{}CYCLE_{}RANK.vtp", base_dir, count, rank);
    fs::create_dir_all(&base_dir)?;
    let mut file = File::create(&filename)?;

    let n = atoms.len();
    let nlocal = atoms.nlocal as usize;

    write!(&mut file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n<PolyData>\n")?;
    writeln!(&mut file, "<Piece NumberOfPoints=\"{}\">", n)?;

    // Points
    write!(&mut file, "<Points><DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">")?;
    for i in 0..n {
        writeln!(&mut file, "{} {} {}", atoms.pos_x[i], atoms.pos_y[i], atoms.pos_z[i])?;
    }
    write!(&mut file, "</DataArray>\n</Points>\n")?;

    // Per-point data arrays
    writeln!(&mut file, "<PointData Scalars=\"\" Vectors=\"\">")?;
    write_vtp_data_array(&mut file, "Float32", "Radius", n, |i| format!("{}", atoms.skin[i]))?;
    write_vtp_data_array(&mut file, "Float32", "Vel_Mag", n, |i| {
        let vmag = (atoms.vel_x[i].powi(2) + atoms.vel_y[i].powi(2) + atoms.vel_z[i].powi(2)).sqrt();
        format!("{}", vmag)
    })?;
    write_vtp_data_array(&mut file, "Int32", "IsGhost", n, |i| {
        format!("{}", if i >= nlocal { 1 } else { 0 })
    })?;
    write_vtp_data_array(&mut file, "Int32", "IsCollision", n, |i| {
        format!("{}", if atoms.is_collision[i] { 1 } else { 0 })
    })?;

    write!(&mut file, "</PointData>\n</Piece>\n</PolyData>\n</VTKFile>\n")?;
    Ok(())
}

// ── Dump output ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn dump_atoms(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    dump_config: Res<DumpConfig>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
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

    if let Err(e) = dump_atoms_inner(&atoms, step, comm.rank(), &input, &dump_config) {
        eprintln!("WARNING: Dump write failed at step {}: {}", step, e);
    }
}

fn dump_atoms_inner(
    atoms: &Atom,
    step: usize,
    rank: i32,
    input: &Input,
    dump_config: &DumpConfig,
) -> std::io::Result<()> {
    let nlocal = atoms.nlocal as usize;
    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("./{}/dump", dir),
        None => "./dump".to_string(),
    };
    fs::create_dir_all(&base_dir)?;

    match dump_config.format.as_str() {
        "binary" => {
            let filename = format!("{}/dump_{}_rank{}.bin", base_dir, step, rank);
            let file = File::create(&filename)?;
            let mut w = BufWriter::new(file);
            w.write_all(&(nlocal as u32).to_le_bytes())?;
            for i in 0..nlocal {
                w.write_all(&atoms.tag[i].to_le_bytes())?;
                w.write_all(&atoms.atom_type[i].to_le_bytes())?;
                w.write_all(&atoms.pos_x[i].to_le_bytes())?;
                w.write_all(&atoms.pos_y[i].to_le_bytes())?;
                w.write_all(&atoms.pos_z[i].to_le_bytes())?;
                w.write_all(&atoms.vel_x[i].to_le_bytes())?;
                w.write_all(&atoms.vel_y[i].to_le_bytes())?;
                w.write_all(&atoms.vel_z[i].to_le_bytes())?;
                w.write_all(&atoms.force_x[i].to_le_bytes())?;
                w.write_all(&atoms.force_y[i].to_le_bytes())?;
                w.write_all(&atoms.force_z[i].to_le_bytes())?;
                w.write_all(&atoms.skin[i].to_le_bytes())?;
            }
        }
        _ => {
            // Default: text/CSV
            let filename = format!("{}/dump_{}_rank{}.csv", base_dir, step, rank);
            let file = File::create(&filename)?;
            let mut w = BufWriter::new(file);
            writeln!(w, "tag,type,x,y,z,vx,vy,vz,fx,fy,fz,radius")?;
            for i in 0..nlocal {
                writeln!(
                    w,
                    "{},{},{},{},{},{},{},{},{},{},{},{}",
                    atoms.tag[i],
                    atoms.atom_type[i],
                    atoms.pos_x[i],
                    atoms.pos_y[i],
                    atoms.pos_z[i],
                    atoms.vel_x[i],
                    atoms.vel_y[i],
                    atoms.vel_z[i],
                    atoms.force_x[i],
                    atoms.force_y[i],
                    atoms.force_z[i],
                    atoms.skin[i],
                )?;
            }
        }
    }
    Ok(())
}

// ── Restart write ───────────────────────────────────────────────────────────

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
    let nlocal = atoms.nlocal as usize;
    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("./{}/restart", dir),
        None => "./restart".to_string(),
    };
    fs::create_dir_all(&base_dir).ok();

    let data = RestartData {
        natoms: atoms.natoms,
        total_cycle: step,
        dt: atoms.dt,
        tag: atoms.tag[..nlocal].to_vec(),
        atom_type: atoms.atom_type[..nlocal].to_vec(),
        pos_x: atoms.pos_x[..nlocal].to_vec(),
        pos_y: atoms.pos_y[..nlocal].to_vec(),
        pos_z: atoms.pos_z[..nlocal].to_vec(),
        vel_x: atoms.vel_x[..nlocal].to_vec(),
        vel_y: atoms.vel_y[..nlocal].to_vec(),
        vel_z: atoms.vel_z[..nlocal].to_vec(),
        omega_x: atoms.omega_x[..nlocal].to_vec(),
        omega_y: atoms.omega_y[..nlocal].to_vec(),
        omega_z: atoms.omega_z[..nlocal].to_vec(),
        force_x: atoms.force_x[..nlocal].to_vec(),
        force_y: atoms.force_y[..nlocal].to_vec(),
        force_z: atoms.force_z[..nlocal].to_vec(),
        torque_x: atoms.torque_x[..nlocal].to_vec(),
        torque_y: atoms.torque_y[..nlocal].to_vec(),
        torque_z: atoms.torque_z[..nlocal].to_vec(),
        ang_mom_x: atoms.ang_mom_x[..nlocal].to_vec(),
        ang_mom_y: atoms.ang_mom_y[..nlocal].to_vec(),
        ang_mom_z: atoms.ang_mom_z[..nlocal].to_vec(),
        quaternion: (0..nlocal)
            .map(|i| {
                let q = atoms.quaternion[i];
                [q.w, q.i, q.j, q.k]
            })
            .collect(),
        mass: atoms.mass[..nlocal].to_vec(),
        skin: atoms.skin[..nlocal].to_vec(),
        atom_data_buffers: registry.pack_all_for_restart(nlocal),
    };

    if let Err(e) = write_restart_inner(&data, &base_dir, step, rank, &restart_config) {
        eprintln!("WARNING: Restart write failed at step {}: {}", step, e);
    }
}

fn write_restart_inner(
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

// ── Restart read ────────────────────────────────────────────────────────────

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
    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("./{}/restart", dir),
        None => "./restart".to_string(),
    };

    // Find the latest restart file for this rank
    let ext = match restart_config.format.as_str() {
        "json" => "json",
        _ => "bin",
    };

    let mut latest_step: Option<usize> = None;
    if let Ok(entries) = fs::read_dir(&base_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let prefix = "restart_".to_string();
            let suffix = format!("_rank{}.{}", rank, ext);
            if name.starts_with(&prefix) && name.ends_with(&suffix) {
                let mid = &name[prefix.len()..name.len() - suffix.len()];
                if let Ok(step) = mid.parse::<usize>() {
                    if latest_step.is_none() || step > latest_step.unwrap() {
                        latest_step = Some(step);
                    }
                }
            }
        }
    }

    let step = match latest_step {
        Some(s) => s,
        None => {
            if comm.rank() == 0 {
                println!("Restart: no restart files found in {}", base_dir);
            }
            return;
        }
    };

    let filename = format!("{}/restart_{}_rank{}.{}", base_dir, step, rank, ext);
    if comm.rank() == 0 {
        println!("Restart: reading from {}", filename);
    }

    let file = File::open(&filename).unwrap_or_else(|e| {
        eprintln!("ERROR: Failed to open restart file '{}': {}", filename, e);
        std::process::exit(1);
    });
    let data: RestartData = match ext {
        "json" => serde_json::from_reader(std::io::BufReader::new(file)).unwrap_or_else(|e| {
            eprintln!("ERROR: Failed to read restart '{}': {}", filename, e);
            std::process::exit(1);
        }),
        _ => bincode::deserialize_from(std::io::BufReader::new(file)).unwrap_or_else(|e| {
            eprintln!("ERROR: Failed to read restart '{}': {}", filename, e);
            std::process::exit(1);
        }),
    };

    let n = data.tag.len();

    // Clear existing atoms and repopulate
    atoms.natoms = data.natoms;
    atoms.nlocal = n as u32;
    atoms.nghost = 0;
    atoms.dt = data.dt;

    atoms.tag = data.tag;
    atoms.atom_type = data.atom_type;
    atoms.origin_index = vec![0; n];
    atoms.is_ghost = vec![false; n];
    atoms.is_collision = vec![false; n];
    atoms.pos_x = data.pos_x;
    atoms.pos_y = data.pos_y;
    atoms.pos_z = data.pos_z;
    atoms.vel_x = data.vel_x;
    atoms.vel_y = data.vel_y;
    atoms.vel_z = data.vel_z;
    atoms.omega_x = data.omega_x;
    atoms.omega_y = data.omega_y;
    atoms.omega_z = data.omega_z;
    atoms.force_x = data.force_x;
    atoms.force_y = data.force_y;
    atoms.force_z = data.force_z;
    atoms.torque_x = data.torque_x;
    atoms.torque_y = data.torque_y;
    atoms.torque_z = data.torque_z;
    atoms.ang_mom_x = data.ang_mom_x;
    atoms.ang_mom_y = data.ang_mom_y;
    atoms.ang_mom_z = data.ang_mom_z;
    atoms.quaternion = data
        .quaternion
        .iter()
        .map(|q| {
            nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                q[0], q[1], q[2], q[3],
            ))
        })
        .collect();
    atoms.mass = data.mass;
    atoms.skin = data.skin;

    // Restore AtomData (DemAtom, etc.) from generic buffers
    if !data.atom_data_buffers.is_empty() {
        registry.unpack_all_from_restart(&data.atom_data_buffers);
    }

    run_state.total_cycle = data.total_cycle;
    if comm.rank() == 0 {
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
}

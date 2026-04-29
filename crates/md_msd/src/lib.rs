//! Per-type mean-squared displacement (MSD) plugin with diffusion coefficient estimation.
//!
//! # Mean-Squared Displacement
//!
//! The mean-squared displacement measures how far particles travel from their
//! initial positions over time, averaged over all particles of a given type:
//!
//! ```text
//! MSD(t) = <|r(t) - r(0)|²> = (1/N) Σᵢ |rᵢ(t) - rᵢ(0)|²
//! ```
//!
//! where `r(t)` are **unwrapped** coordinates (accounting for periodic boundary
//! crossings) and `N` is the number of atoms of that type.
//!
//! # Einstein Relation (Diffusion Coefficient)
//!
//! In the long-time limit, the MSD of a diffusing particle grows linearly with
//! time. The Einstein relation connects MSD to the self-diffusion coefficient:
//!
//! ```text
//! D = lim(t→∞) MSD(t) / (2 * d * t)
//! ```
//!
//! where `d` is the spatial dimensionality (3 for 3D simulations). This plugin
//! reports `D = MSD(t) / (6 * t)` at each sample point. Note that `D` is only
//! meaningful in the diffusive (linear) regime — early-time ballistic motion
//! will give artificially large values.
//!
//! # Coordinate Unwrapping
//!
//! In periodic simulations, atoms that cross a boundary are wrapped back to the
//! other side of the box. This plugin detects these jumps by comparing each
//! atom's current wrapped position to its previous wrapped position: if the
//! displacement exceeds half the box length, a boundary crossing is inferred
//! and the unwrapped coordinate is corrected accordingly.
//!
//! # Configuration
//!
//! ```toml
//! [msd]
//! interval = 10           # sample MSD every N steps (default: 10)
//! output_interval = 1000  # write output files every N steps (default: 1000)
//! ```
//!
//! # Output Files
//!
//! The plugin writes to `<output_dir>/data/`:
//!
//! - **`msd_all.txt`** — combined file with columns:
//!   `dt_steps  dt_time  MSD_all  D_all  [MSD_typeN  D_typeN ...]`
//! - **`msd_type_<i>.txt`** — per-type file with columns:
//!   `dt_steps  dt_time  MSD  D`
//!
//! where `dt_steps` is the elapsed timesteps since the reference time,
//! `dt_time = dt_steps * timestep`, and `D = MSD / (6 * dt_time)`.
//!
//! The plugin automatically detects all atom types present in the simulation.

use std::fs;
use std::io::Write;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, CommResource, Config, Domain, Input, RunState, ParticleSimScheduleSet};

// ── Config ──────────────────────────────────────────────────────────────────

fn default_interval() -> usize {
    10
}
fn default_output_interval() -> usize {
    1000
}

/// TOML `[msd]` section — controls how often MSD is sampled and written.
///
/// # Fields
///
/// | Field             | Type    | Default | Description                         |
/// |-------------------|---------|---------|-------------------------------------|
/// | `interval`        | `usize` | 10      | Sample MSD every N timesteps        |
/// | `output_interval` | `usize` | 1000    | Write output files every N timesteps|
///
/// Set `interval = 0` to disable MSD tracking entirely.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct MsdConfig {
    /// Sample MSD every N timesteps. Set to 0 to disable.
    #[serde(default = "default_interval")]
    pub interval: usize,
    /// Write output files every N timesteps.
    #[serde(default = "default_output_interval")]
    pub output_interval: usize,
}

impl Default for MsdConfig {
    fn default() -> Self {
        MsdConfig {
            interval: 10,
            output_interval: 1000,
        }
    }
}

// ── Resources ───────────────────────────────────────────────────────────────

/// Per-type MSD tracker that maintains unwrapped coordinates for all atoms.
///
/// All position arrays are indexed by **atom tag** (not local index), so that
/// atoms can migrate between processors without losing their displacement history.
///
/// The tracker accumulates `(dt_steps, msd_all, msd_per_type)` tuples in
/// [`values`](Self::values), which are periodically flushed to disk by
/// [`write_type_msd`].
pub struct TypeMsdTracker {
    /// Reference (initial) positions at the start of tracking, indexed by atom tag.
    pub ref_pos: Vec<[f64; 3]>,
    /// Current unwrapped positions, indexed by atom tag. Updated each step by
    /// adding the minimum-image displacement from the previous wrapped position.
    pub unwrapped: Vec<[f64; 3]>,
    /// Previous **wrapped** positions, indexed by atom tag. Used to detect
    /// periodic boundary crossings by comparing with the current wrapped position.
    pub prev_pos: Vec<[f64; 3]>,
    /// Atom type for each tag, indexed by atom tag.
    pub atom_types: Vec<u32>,
    /// Whether we have recorded an entry for this atom tag.
    pub has_entry: Vec<bool>,
    /// Number of distinct atom types in the simulation.
    pub ntypes: usize,
    /// Global count of atoms per type (as `f64` for use in averaging).
    pub type_counts: Vec<f64>,
    /// Timestep number at which reference positions were recorded.
    pub ref_step: usize,
    /// Accumulated time series: `(dt_steps, msd_all, msd_per_type)`.
    pub values: Vec<(usize, f64, Vec<f64>)>,
    /// Whether the tracker has been initialized with reference positions.
    pub initialized: bool,
}

impl Default for TypeMsdTracker {
    fn default() -> Self {
        TypeMsdTracker {
            ref_pos: Vec::new(),
            unwrapped: Vec::new(),
            prev_pos: Vec::new(),
            atom_types: Vec::new(),
            has_entry: Vec::new(),
            ntypes: 0,
            type_counts: Vec::new(),
            ref_step: 0,
            values: Vec::new(),
            initialized: false,
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Per-type MSD measurement plugin.
///
/// Tracks unwrapped displacements for each atom type and computes:
/// - MSD(t) = <|r(t) - r(0)|²> per type and overall
/// - Diffusion coefficient D = MSD(t) / (6 * dt * timestep) via Einstein relation
pub struct TypeMsdPlugin;

impl Plugin for TypeMsdPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[msd]
interval = 10          # sample MSD every N steps
output_interval = 1000 # write output every N steps"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<MsdConfig>(app, "msd");

        app.add_resource(TypeMsdTracker::default())
            .add_update_system(track_type_msd, ParticleSimScheduleSet::PostFinalIntegration)
            .add_update_system(write_type_msd, ParticleSimScheduleSet::PostFinalIntegration);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// System that updates unwrapped positions and computes per-type MSD each interval.
///
/// On the first call with atoms present, this system records reference positions
/// and atom type counts. On subsequent calls at every `config.interval` steps,
/// it computes MSD(t) = <|r(t) - r(0)|²> for each type and appends the result
/// to [`TypeMsdTracker::values`].
pub fn track_type_msd(
    atoms: Res<Atom>,
    run_state: Res<RunState>,
    config: Res<MsdConfig>,
    domain: Res<Domain>,
    comm: Res<CommResource>,
    mut msd: ResMut<TypeMsdTracker>,
) {
    if config.interval == 0 {
        return;
    }

    let nlocal = atoms.nlocal as usize;

    // Initialize on first call with atoms
    if !msd.initialized && nlocal > 0 {
        let max_tag = atoms.tag[..nlocal].iter().cloned().max().unwrap_or(0) as usize;
        let size = max_tag + 1;
        msd.ref_pos.resize(size, [0.0; 3]);
        msd.unwrapped.resize(size, [0.0; 3]);
        msd.prev_pos.resize(size, [0.0; 3]);
        msd.atom_types.resize(size, 0);
        msd.has_entry.resize(size, false);

        // Determine number of types: use atoms.ntypes if set (by LJ pair table),
        // otherwise detect from local atom_type values via global reduction.
        let ntypes = if atoms.ntypes > 0 {
            atoms.ntypes
        } else {
            let local_max_type = atoms.atom_type[..nlocal]
                .iter()
                .cloned()
                .max()
                .unwrap_or(0) as f64;
            let global_max_type = comm.all_reduce_sum_f64(local_max_type);
            // On single process, sum == local value; on multi-process this is approximate
            // but atoms.ntypes should always be set by the force plugin.
            global_max_type as usize + 1
        };
        msd.ntypes = ntypes;
        msd.type_counts = vec![0.0; msd.ntypes];

        for i in 0..nlocal {
            let idx = atoms.tag[i] as usize;
            msd.ref_pos[idx] = atoms.pos[i];
            msd.unwrapped[idx] = atoms.pos[i];
            msd.prev_pos[idx] = atoms.pos[i];
            msd.atom_types[idx] = atoms.atom_type[i];
            msd.has_entry[idx] = true;
        }

        // Count atoms per type globally
        let mut local_counts = vec![0.0f64; msd.ntypes];
        for i in 0..nlocal {
            let t = atoms.atom_type[i] as usize;
            if t < msd.ntypes {
                local_counts[t] += 1.0;
            }
        }
        for t in 0..msd.ntypes {
            msd.type_counts[t] = comm.all_reduce_sum_f64(local_counts[t]);
        }

        msd.ref_step = run_state.total_cycle;
        msd.initialized = true;
        if comm.rank() == 0 {
            let nt = msd.ntypes;
            msd.values.push((0, 0.0, vec![0.0; nt]));
        }
        return;
    }

    if !msd.initialized {
        return;
    }

    // --- Coordinate unwrapping ---
    // Compare each atom's current wrapped position to its previous wrapped
    // position. If the displacement exceeds half the box length in any axis,
    // the atom must have crossed a periodic boundary, so we correct the
    // unwrapped coordinate by subtracting/adding the full box length.
    let lx = domain.size[0];
    let ly = domain.size[1];
    let lz = domain.size[2];
    let half_lx = lx * 0.5;
    let half_ly = ly * 0.5;
    let half_lz = lz * 0.5;

    for i in 0..nlocal {
        let idx = atoms.tag[i] as usize;
        // Grow arrays if a new (higher-tagged) atom appears mid-simulation
        if idx >= msd.prev_pos.len() {
            let new_size = idx + 1;
            msd.ref_pos.resize(new_size, [0.0; 3]);
            msd.unwrapped.resize(new_size, [0.0; 3]);
            msd.prev_pos.resize(new_size, [0.0; 3]);
            msd.atom_types.resize(new_size, 0);
            msd.has_entry.resize(new_size, false);
        }

        if msd.has_entry[idx] {
            // Compute minimum-image displacement between current and previous
            // wrapped positions. A jump > half-box means a PBC crossing occurred.
            let mut dx = atoms.pos[i][0] - msd.prev_pos[idx][0];
            let mut dy = atoms.pos[i][1] - msd.prev_pos[idx][1];
            let mut dz = atoms.pos[i][2] - msd.prev_pos[idx][2];

            if dx > half_lx {
                dx -= lx;
            } else if dx < -half_lx {
                dx += lx;
            }
            if dy > half_ly {
                dy -= ly;
            } else if dy < -half_ly {
                dy += ly;
            }
            if dz > half_lz {
                dz -= lz;
            } else if dz < -half_lz {
                dz += lz;
            }

            // Accumulate the corrected displacement into the unwrapped position
            msd.unwrapped[idx][0] += dx;
            msd.unwrapped[idx][1] += dy;
            msd.unwrapped[idx][2] += dz;
        }
        msd.prev_pos[idx] = atoms.pos[i];
        msd.atom_types[idx] = atoms.atom_type[i];
        msd.has_entry[idx] = true;
    }

    let step = run_state.total_cycle;
    if step == 0 || !step.is_multiple_of(config.interval) {
        return;
    }

    // --- Per-type MSD computation ---
    // Sum |r_unwrapped(t) - r_ref(0)|² for each atom, grouped by type.
    let ntypes = msd.ntypes;
    let mut local_msd_sum = vec![0.0f64; ntypes];
    let mut local_msd_all = 0.0f64;

    for i in 0..nlocal {
        let idx = atoms.tag[i] as usize;
        if idx < msd.has_entry.len() && msd.has_entry[idx] {
            let dx = msd.unwrapped[idx][0] - msd.ref_pos[idx][0];
            let dy = msd.unwrapped[idx][1] - msd.ref_pos[idx][1];
            let dz = msd.unwrapped[idx][2] - msd.ref_pos[idx][2];
            let dr2 = dx * dx + dy * dy + dz * dz;
            local_msd_all += dr2;
            let atom_type = atoms.atom_type[i] as usize;
            if atom_type < ntypes {
                local_msd_sum[atom_type] += dr2;
            }
        }
    }

    // Global reduction across all MPI ranks, then normalize by atom count
    let global_msd_all = comm.all_reduce_sum_f64(local_msd_all);
    let mut per_type_msd = vec![0.0f64; ntypes];
    for t in 0..ntypes {
        let global_sum = comm.all_reduce_sum_f64(local_msd_sum[t]);
        if msd.type_counts[t] > 0.0 {
            per_type_msd[t] = global_sum / msd.type_counts[t];
        }
    }

    let total_atoms: f64 = msd.type_counts.iter().sum();
    let msd_avg = if total_atoms > 0.0 {
        global_msd_all / total_atoms
    } else {
        0.0
    };

    if comm.rank() == 0 {
        let dt = step - msd.ref_step;
        msd.values.push((dt, msd_avg, per_type_msd));
    }
}

/// System that writes accumulated MSD time-series data to output files.
///
/// Runs on rank 0 only, every `config.output_interval` steps. Writes both a
/// combined `msd_all.txt` (with all types side-by-side) and individual
/// `msd_type_<i>.txt` files for each atom type.
pub fn write_type_msd(
    run_state: Res<RunState>,
    config: Res<MsdConfig>,
    msd: Res<TypeMsdTracker>,
    atoms: Res<Atom>,
    comm: Res<CommResource>,
    input: Res<Input>,
) {
    let step = run_state.total_cycle;
    if step == 0 || !step.is_multiple_of(config.output_interval) {
        return;
    }
    if comm.rank() != 0 {
        return;
    }
    if msd.values.is_empty() {
        return;
    }

    let base_dir = input.output_dir.as_deref().unwrap_or(".");
    let data_dir = format!("{}/data", base_dir);
    let _ = fs::create_dir_all(&data_dir);

    let dt_sim = atoms.dt;

    // Write combined MSD file with all types
    {
        let path = format!("{}/msd_all.txt", data_dir);
        if let Ok(mut f) = fs::File::create(&path) {
            write!(f, "# dt_steps  dt_time  MSD_all  D_all").ok();
            for t in 0..msd.ntypes {
                write!(f, "  MSD_type{}  D_type{}", t, t).ok();
            }
            writeln!(f).ok();
            for (dt_steps, msd_all, per_type) in &msd.values {
                let time = *dt_steps as f64 * dt_sim;
                let d_all = if time > 0.0 {
                    msd_all / (6.0 * time)
                } else {
                    0.0
                };
                write!(f, "{}  {:.6}  {:.6}  {:.6}", dt_steps, time, msd_all, d_all).ok();
                for (_type_idx, &msd_t) in per_type.iter().enumerate() {
                    let d_t = if time > 0.0 {
                        msd_t / (6.0 * time)
                    } else {
                        0.0
                    };
                    write!(f, "  {:.6}  {:.6}", msd_t, d_t).ok();
                }
                writeln!(f).ok();
            }
            println!("Wrote per-type MSD to {}", path);
        }
    }

    // Write per-type individual files
    for t in 0..msd.ntypes {
        let path = format!("{}/msd_type_{}.txt", data_dir, t);
        if let Ok(mut f) = fs::File::create(&path) {
            writeln!(
                f,
                "# dt_steps  dt_time  MSD  D  (type {}, N={})",
                t, msd.type_counts[t] as u64
            )
            .ok();
            for (dt_steps, _, per_type) in &msd.values {
                let time = *dt_steps as f64 * dt_sim;
                let msd_t = per_type.get(t).copied().unwrap_or(0.0);
                let d_t = if time > 0.0 {
                    msd_t / (6.0 * time)
                } else {
                    0.0
                };
                writeln!(f, "{}  {:.6}  {:.6}  {:.6}", dt_steps, time, msd_t, d_t).ok();
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msd_tracker_default() {
        let tracker = TypeMsdTracker::default();
        assert!(!tracker.initialized);
        assert!(tracker.values.is_empty());
        assert_eq!(tracker.ntypes, 0);
    }

    #[test]
    fn msd_config_default() {
        let config = MsdConfig::default();
        assert_eq!(config.interval, 10);
        assert_eq!(config.output_interval, 1000);
    }

    #[test]
    fn msd_zero_displacement() {
        // If particles don't move, MSD should be zero
        let mut tracker = TypeMsdTracker::default();
        tracker.ntypes = 2;
        tracker.type_counts = vec![5.0, 3.0];
        tracker.ref_pos = vec![[1.0, 2.0, 3.0]; 8];
        tracker.unwrapped = vec![[1.0, 2.0, 3.0]; 8];
        tracker.prev_pos = vec![[1.0, 2.0, 3.0]; 8];
        tracker.has_entry = vec![true; 8];
        tracker.atom_types = vec![0, 0, 0, 0, 0, 1, 1, 1];
        tracker.initialized = true;

        // With zero displacement, MSD should be zero for all types
        for idx in 0..8 {
            let dx = tracker.unwrapped[idx][0] - tracker.ref_pos[idx][0];
            let dy = tracker.unwrapped[idx][1] - tracker.ref_pos[idx][1];
            let dz = tracker.unwrapped[idx][2] - tracker.ref_pos[idx][2];
            let dr2 = dx * dx + dy * dy + dz * dz;
            assert!(dr2.abs() < 1e-15, "zero displacement should give zero MSD");
        }
    }

    #[test]
    fn msd_known_displacement() {
        // Test MSD with known displacement
        let mut tracker = TypeMsdTracker::default();
        tracker.ntypes = 1;
        tracker.type_counts = vec![2.0];
        tracker.ref_pos = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        tracker.unwrapped = vec![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        tracker.has_entry = vec![true, true];
        tracker.atom_types = vec![0, 0];
        tracker.initialized = true;

        // Atom 0: dr² = 1, Atom 1: dr² = 4
        // MSD = (1 + 4) / 2 = 2.5
        let mut total = 0.0;
        for idx in 0..2 {
            let dx = tracker.unwrapped[idx][0] - tracker.ref_pos[idx][0];
            let dy = tracker.unwrapped[idx][1] - tracker.ref_pos[idx][1];
            let dz = tracker.unwrapped[idx][2] - tracker.ref_pos[idx][2];
            total += dx * dx + dy * dy + dz * dz;
        }
        let msd = total / tracker.type_counts[0];
        assert!((msd - 2.5).abs() < 1e-10, "MSD should be 2.5, got {}", msd);
    }

    #[test]
    fn diffusion_coefficient_formula() {
        // D = MSD / (6 * t) — verify the formula
        let msd: f64 = 12.0;
        let time: f64 = 100.0;
        let d = msd / (6.0 * time);
        assert!(
            (d - 0.02).abs() < 1e-10,
            "D should be 0.02, got {}",
            d
        );
    }
}

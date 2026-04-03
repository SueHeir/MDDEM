//! Box deformation fix for MDDEM.
//!
//! Continuously modifies simulation box boundaries during a run, analogous to
//! LAMMPS `fix deform`. Supports per-axis independent control with three styles:
//!
//! - **erate** — engineering strain rate: `L(t) = L0 * (1 + rate * dt * step)`
//! - **vel** — constant velocity on box faces: `L(t) = L0 + velocity * dt * step`
//! - **final** — linear ramp to a target value over the run
//!
//! Atom positions are remapped proportionally (affine transformation) when the
//! box changes, ensuring particles stay inside the domain.
//!
//! # Configuration
//!
//! ```toml
//! [deform]
//! # Constant engineering strain rate on z-axis (compression = negative)
//! z = { style = "erate", rate = -0.001 }
//!
//! # Constant velocity on x-axis faces
//! # x = { style = "vel", velocity = 0.01 }
//!
//! # Ramp y-axis to target bounds over the run
//! # y = { style = "final", lo = 0.0, hi = 0.02 }
//!
//! # Remap atom positions (default: true)
//! remap = true
//! ```
//!
//! # Scheduling
//!
//! The deform system runs at [`ParticleSimScheduleSet::PreInitialIntegration`], updating
//! domain bounds and remapping atoms before the Verlet position update.

use sim_app::prelude::*;
use mddem_core::{Atom, CommResource, Config, Domain, ParticleSimScheduleSet, ScheduleSetupSet, StageOverrides};
use mddem_neighbor::Neighbor;
use sim_scheduler::prelude::*;
use serde::Deserialize;

// ── TOML config structs ─────────────────────────────────────────────────────

/// Serde default for boolean fields that should be `true` when omitted.
fn default_true() -> bool {
    true
}

/// Human-readable axis names indexed by dimension (0=x, 1=y, 2=z).
const AXIS_NAMES: [&str; 3] = ["x", "y", "z"];

/// Per-axis deformation definition parsed from TOML.
///
/// Each axis (`x`, `y`, `z`) can independently specify a deformation style.
/// Only the fields relevant to the chosen style are required; unused fields
/// should be omitted.
///
/// ```toml
/// z = { style = "erate", rate = -0.001 }
/// x = { style = "vel", velocity = 0.01 }
/// y = { style = "final", lo = 0.0, hi = 0.02 }
/// ```
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct AxisDeformDef {
    /// Deformation style: `"erate"`, `"vel"`, or `"final"`.
    pub style: String,
    /// Engineering strain rate (required for `"erate"` style). Negative = compression.
    #[serde(default)]
    pub rate: Option<f64>,
    /// Constant velocity applied to box faces (required for `"vel"` style).
    /// Positive = expansion, negative = compression.
    #[serde(default)]
    pub velocity: Option<f64>,
    /// Target lower bound (required for `"final"` style).
    #[serde(default)]
    pub lo: Option<f64>,
    /// Target upper bound (required for `"final"` style).
    #[serde(default)]
    pub hi: Option<f64>,
}

/// Top-level `[deform]` TOML configuration section.
///
/// Each axis can be independently controlled (or omitted to leave it unchanged).
/// When `remap` is true (the default), atom positions are affinely scaled to
/// follow the box deformation, preserving their relative positions within the
/// domain.
///
/// # Example
///
/// ```toml
/// [deform]
/// z = { style = "erate", rate = -0.001 }  # uniaxial compression on z
/// remap = true
/// ```
#[derive(Deserialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct DeformConfig {
    /// X-axis deformation definition (omit to leave x unchanged).
    #[serde(default)]
    pub x: Option<AxisDeformDef>,
    /// Y-axis deformation definition (omit to leave y unchanged).
    #[serde(default)]
    pub y: Option<AxisDeformDef>,
    /// Z-axis deformation definition (omit to leave z unchanged).
    #[serde(default)]
    pub z: Option<AxisDeformDef>,
    /// Whether to remap atom positions proportionally when the box changes.
    /// Default: `true`.
    #[serde(default = "default_true")]
    pub remap: bool,
}

// ── Runtime types ───────────────────────────────────────────────────────────

/// Deformation style for a single axis.
#[derive(Clone, Debug)]
pub enum DeformStyle {
    /// Engineering strain rate: L(t) = L0 * (1 + rate * dt * step)
    Erate { rate: f64 },
    /// Constant velocity on box faces: L(t) = L0 + velocity * dt * step
    Vel { velocity: f64 },
    /// Linear ramp to target lo/hi over the run duration.
    Final { target_lo: f64, target_hi: f64 },
}

/// Per-axis deformation state.
#[derive(Clone, Debug)]
pub struct AxisDeform {
    pub style: DeformStyle,
    /// Initial lower bound at start of deformation.
    pub lo_0: f64,
    /// Initial upper bound at start of deformation.
    pub hi_0: f64,
}

/// Runtime resource holding the current deformation state for all axes.
///
/// Registered as a resource by [`DeformPlugin`]. The setup system reinitializes
/// this each stage from the (possibly overridden) `[deform]` config, and the
/// update system mutates `step` and domain bounds every timestep.
pub struct DeformState {
    /// Per-axis deform (`None` = axis not deforming). Indices: 0=x, 1=y, 2=z.
    pub axes: [Option<AxisDeform>; 3],
    /// Whether to remap atom positions affinely when the box changes.
    pub remap: bool,
    /// Number of timesteps elapsed since the current stage's deformation started.
    pub step: usize,
    /// Whether the deform state has been initialized with domain bounds.
    pub initialized: bool,
}

impl DeformState {
    /// Returns `true` if at least one axis has an active deformation definition.
    pub fn has_any(&self) -> bool {
        self.axes.iter().any(|a| a.is_some())
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Plugin that enables simulation box deformation during a run.
///
/// Registers the [`DeformState`] resource and two systems:
/// - **Setup** ([`ScheduleSetupSet::PostSetup`]): reads the `[deform]` config
///   (including per-stage overrides) and initializes axis definitions.
/// - **Update** ([`ParticleSimScheduleSet::PreInitialIntegration`]): applies the
///   deformation each timestep before the Verlet position update.
///
/// Add this plugin to your app to enable `[deform]` TOML configuration:
///
/// ```ignore
/// app.add_plugin(DeformPlugin);
/// ```
pub struct DeformPlugin;

impl Plugin for DeformPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# [deform]
# Per-axis box deformation (compression, expansion, strain control)
# z = { style = "erate", rate = -0.001 }    # engineering strain rate
# x = { style = "vel", velocity = 0.01 }    # constant velocity
# y = { style = "final", lo = 0.0, hi = 0.02 }  # ramp to target
# remap = true  # remap atom positions (default: true)"#,
        )
    }

    fn build(&self, app: &mut App) {
        // Register a default (empty) DeformState. The setup system re-reads
        // the merged stage-aware config each stage, so we don't parse axes here.
        Config::load::<DeformConfig>(app, "deform");

        let state = DeformState {
            axes: [None, None, None],
            remap: true,
            step: 0,
            initialized: false,
        };

        // Always register systems — the setup system reads per-stage config and
        // populates axes (or clears them) accordingly. The update system early-
        // returns when no axes are active.
        app.add_resource(state)
            .add_setup_system(setup_deform, ScheduleSetupSet::PostSetup)
            .add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
    }
}

/// Parse an optional axis definition from config into a runtime [`AxisDeform`].
///
/// Returns `None` if `def` is `None` (axis not configured). The `lo_0`/`hi_0`
/// fields are initialized to zero here and filled in by the setup system once
/// the current domain bounds are known.
///
/// # Panics
///
/// - If `style` is `"erate"` but `rate` is missing.
/// - If `style` is `"vel"` but `velocity` is missing.
/// - If `style` is `"final"` but `lo` or `hi` is missing.
/// - If `style` is an unrecognized string.
fn parse_axis_def(def: &Option<AxisDeformDef>, axis_name: &str) -> Option<AxisDeform> {
    let def = def.as_ref()?;
    let style = match def.style.as_str() {
        "erate" => {
            let rate = def.rate.unwrap_or_else(|| {
                panic!("[deform] {axis_name}: style 'erate' requires 'rate' field");
            });
            DeformStyle::Erate { rate }
        }
        "vel" => {
            let velocity = def.velocity.unwrap_or_else(|| {
                panic!("[deform] {axis_name}: style 'vel' requires 'velocity' field");
            });
            DeformStyle::Vel { velocity }
        }
        "final" => {
            let lo = def.lo.unwrap_or_else(|| {
                panic!("[deform] {axis_name}: style 'final' requires 'lo' field");
            });
            let hi = def.hi.unwrap_or_else(|| {
                panic!("[deform] {axis_name}: style 'final' requires 'hi' field");
            });
            DeformStyle::Final {
                target_lo: lo,
                target_hi: hi,
            }
        }
        other => {
            panic!("[deform] {axis_name}: unknown style '{other}'. Use 'erate', 'vel', or 'final'.");
        }
    };

    Some(AxisDeform {
        style,
        lo_0: 0.0,
        hi_0: 0.0,
    })
}

// ── Setup system ────────────────────────────────────────────────────────────

/// Re-read deformation config from stage overrides and initialize state.
///
/// This runs at the start of every stage, so per-stage `[deform]` overrides
/// inside `[[run]]` blocks take effect. When a stage has no `[deform]`
/// section, all axes are cleared so deformation doesn't persist from a
/// previous stage.
fn setup_deform(
    mut state: ResMut<DeformState>,
    domain: Res<Domain>,
    comm: Res<CommResource>,
    stage_overrides: Res<StageOverrides>,
) {
    // Re-read config from the merged (global + current stage) config table.
    let config: DeformConfig = Config::load_stage_aware(&stage_overrides, "deform");

    // Parse axis definitions fresh from this stage's config.
    state.axes = [
        parse_axis_def(&config.x, "x"),
        parse_axis_def(&config.y, "y"),
        parse_axis_def(&config.z, "z"),
    ];
    state.remap = config.remap;

    // Set initial bounds from current domain and reset step counter.
    for (dim, axis) in state.axes.iter_mut().enumerate() {
        if let Some(ref mut ax) = axis {
            ax.lo_0 = domain.boundaries_low[dim];
            ax.hi_0 = domain.boundaries_high[dim];
        }
    }
    state.initialized = true;
    state.step = 0;

    if comm.rank() == 0 && state.has_any() {
        for (dim, axis) in state.axes.iter().enumerate() {
            if let Some(ref ax) = axis {
                let range = ax.hi_0 - ax.lo_0;
                match &ax.style {
                    DeformStyle::Erate { rate } => {
                        println!(
                            "Deform {}: erate rate={:.6e}, L0={:.6e} [{:.6e}, {:.6e}]",
                            AXIS_NAMES[dim], rate, range, ax.lo_0, ax.hi_0
                        );
                    }
                    DeformStyle::Vel { velocity } => {
                        println!(
                            "Deform {}: vel velocity={:.6e}, L0={:.6e} [{:.6e}, {:.6e}]",
                            AXIS_NAMES[dim], velocity, range, ax.lo_0, ax.hi_0
                        );
                    }
                    DeformStyle::Final {
                        target_lo,
                        target_hi,
                    } => {
                        println!(
                            "Deform {}: final [{:.6e}, {:.6e}] -> [{:.6e}, {:.6e}]",
                            AXIS_NAMES[dim], ax.lo_0, ax.hi_0, target_lo, target_hi
                        );
                    }
                }
            }
        }
    }
}

// ── Update system ───────────────────────────────────────────────────────────

/// Apply box deformation each timestep. Runs at [`ParticleSimScheduleSet::PreInitialIntegration`].
///
/// 1. Compute new domain bounds based on deformation style and current step.
/// 2. Remap atom positions proportionally (affine transform).
/// 3. Update domain resource (bounds, size, volume, sub-domain).
/// 4. Signal neighbor rebuild needed by clearing last-build positions.
///
/// # Panics
///
/// Panics if any axis collapses to zero or negative size (e.g., excessive
/// compression rate).
fn apply_deform(
    mut atoms: ResMut<Atom>,
    mut domain: ResMut<Domain>,
    mut state: ResMut<DeformState>,
    mut neighbor: ResMut<Neighbor>,
    run_state: Res<mddem_core::RunState>,
) {
    if !state.initialized || !state.has_any() {
        return;
    }

    state.step += 1;
    let dt = atoms.dt;
    let step = state.step;
    let nlocal = atoms.nlocal as usize;

    // Total steps for the current stage, needed by "final" style to compute
    // interpolation fraction.  RunState tracks per-stage progress as parallel
    // vectors: `cycle_count[i]` = steps completed so far in stage i,
    // `cycle_remaining[i]` = steps left. Their sum is the stage's total steps.
    let total_steps = if !run_state.cycle_remaining.is_empty() {
        let stage_idx = run_state.cycle_count.len().saturating_sub(1);
        if stage_idx < run_state.cycle_remaining.len() {
            (run_state.cycle_count[stage_idx] + run_state.cycle_remaining[stage_idx]) as usize
        } else {
            1
        }
    } else {
        1
    };

    let mut box_changed = false;

    for dim in 0..3 {
        let axis = match &state.axes[dim] {
            Some(ax) => ax,
            None => continue,
        };

        let old_lo = domain.boundaries_low[dim];
        let old_hi = domain.boundaries_high[dim];
        let old_size = old_hi - old_lo;

        let lo_0 = axis.lo_0;
        let hi_0 = axis.hi_0;
        let l0 = hi_0 - lo_0;

        // Compute new bounds based on style
        let (new_lo, new_hi) = match &axis.style {
            DeformStyle::Erate { rate } => {
                // L(t) = L0 * (1 + rate * dt * step)
                // Symmetric expansion/contraction about center
                let scale = 1.0 + rate * dt * step as f64;
                let new_l = l0 * scale;
                let center = (lo_0 + hi_0) * 0.5;
                (center - new_l * 0.5, center + new_l * 0.5)
            }
            DeformStyle::Vel { velocity } => {
                // L(t) = L0 + velocity * dt * step
                // Symmetric: each face moves by velocity/2 * dt * step
                let delta = velocity * dt * step as f64;
                let new_l = l0 + delta;
                let center = (lo_0 + hi_0) * 0.5;
                (center - new_l * 0.5, center + new_l * 0.5)
            }
            DeformStyle::Final {
                target_lo,
                target_hi,
            } => {
                // Linear interpolation from initial to target
                let frac = if total_steps > 0 {
                    (step as f64) / (total_steps as f64)
                } else {
                    1.0
                };
                let frac = frac.min(1.0);
                let new_lo = lo_0 + (target_lo - lo_0) * frac;
                let new_hi = hi_0 + (target_hi - hi_0) * frac;
                (new_lo, new_hi)
            }
        };

        let new_size = new_hi - new_lo;

        // Sanity: don't allow box to collapse to zero or negative
        if new_size <= 0.0 {
            panic!(
                "[deform] axis {} collapsed at step {}: new_lo={}, new_hi={}",
                AXIS_NAMES[dim],
                step,
                new_lo,
                new_hi
            );
        }

        // Remap atom positions proportionally
        if state.remap && old_size > 0.0 {
            let scale = new_size / old_size;
            let new_center = (new_lo + new_hi) * 0.5;
            let old_center = (old_lo + old_hi) * 0.5;

            for i in 0..nlocal {
                // Affine transform: shift to old center, scale, shift to new center
                atoms.pos[i][dim] =
                    new_center + (atoms.pos[i][dim] - old_center) * scale;
            }
        }

        // Update domain bounds
        domain.boundaries_low[dim] = new_lo;
        domain.boundaries_high[dim] = new_hi;
        domain.size[dim] = new_size;

        // Update sub-domain bounds (single-processor: same as global)
        // For multi-proc, this is a simplification; full MPI would need
        // re-decomposition, but for single-proc it's correct.
        domain.sub_domain_low[dim] = new_lo;
        domain.sub_domain_high[dim] = new_hi;
        domain.sub_length[dim] = new_size;

        box_changed = true;
    }

    if box_changed {
        // Update volume
        domain.volume = domain.size[0] * domain.size[1] * domain.size[2];

        // Force neighbor rebuild by clearing last build positions
        neighbor.last_build_pos.clear();
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::RunState;
    use mddem_test_utils::make_atoms;

    fn make_domain(lo: [f64; 3], hi: [f64; 3]) -> Domain {
        let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        Domain {
            boundaries_low: lo,
            boundaries_high: hi,
            sub_domain_low: lo,
            sub_domain_high: hi,
            sub_length: size,
            size,
            volume: size[0] * size[1] * size[2],
            boundary_type: Default::default(),
            shrink_wrap_padding: 0.0,
            bounds_changed: false,
            ghost_cutoff: 0.0,
        }
    }

    fn make_neighbor() -> Neighbor {
        Neighbor::new()
    }

    fn make_run_state(total_steps: u32) -> RunState {
        let mut rs = RunState::new();
        rs.cycle_count.push(0);
        rs.cycle_remaining.push(total_steps);
        rs
    }

    #[test]
    fn erate_shrinks_box_symmetrically() {
        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mut atoms = make_atoms(0);
        atoms.dt = 0.001;
        let neighbor = make_neighbor();
        let run_state = make_run_state(1000);

        let state = DeformState {
            axes: [
                None,
                None,
                Some(AxisDeform {
                    style: DeformStyle::Erate { rate: -1.0 },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
            ],
            remap: true,
            step: 0,
            initialized: true,
        };

        // After 1 step with rate=-1.0, dt=0.001:
        // scale = 1 + (-1.0) * 0.001 * 1 = 0.999
        // new L = 1.0 * 0.999 = 0.999
        // center = 0.5 => new_lo = 0.5 - 0.4995 = 0.0005, new_hi = 0.5 + 0.4995 = 0.9995
        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let d = app.get_resource_ref::<Domain>().unwrap();
        assert!((d.boundaries_low[2] - 0.0005).abs() < 1e-12);
        assert!((d.boundaries_high[2] - 0.9995).abs() < 1e-12);
        assert!((d.size[2] - 0.999).abs() < 1e-12);
        // x and y should be unchanged
        assert!((d.boundaries_low[0]).abs() < 1e-12);
        assert!((d.boundaries_high[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn vel_expands_box() {
        let mut atoms = make_atoms(0);
        atoms.dt = 0.01;
        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let neighbor = make_neighbor();
        let run_state = make_run_state(100);

        let state = DeformState {
            axes: [
                Some(AxisDeform {
                    style: DeformStyle::Vel { velocity: 1.0 },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
                None,
                None,
            ],
            remap: true,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        // After 1 step: delta = 1.0 * 0.01 * 1 = 0.01
        // new_l = 1.0 + 0.01 = 1.01, center = 0.5
        // new_lo = 0.5 - 0.505 = -0.005, new_hi = 0.5 + 0.505 = 1.005
        let d = app.get_resource_ref::<Domain>().unwrap();
        assert!((d.size[0] - 1.01).abs() < 1e-12);
    }

    #[test]
    fn final_ramps_to_target() {
        let mut atoms = make_atoms(0);
        atoms.dt = 0.001;
        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let neighbor = make_neighbor();
        let run_state = make_run_state(10); // 10 total steps

        let state = DeformState {
            axes: [
                None,
                Some(AxisDeform {
                    style: DeformStyle::Final {
                        target_lo: 0.1,
                        target_hi: 0.9,
                    },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
                None,
            ],
            remap: true,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();

        // Run 10 steps
        for _ in 0..10 {
            app.run();
        }

        let d = app.get_resource_ref::<Domain>().unwrap();
        // After 10 steps of 10 total: frac = 1.0
        // new_lo = 0.0 + (0.1 - 0.0) * 1.0 = 0.1
        // new_hi = 1.0 + (0.9 - 1.0) * 1.0 = 0.9
        assert!((d.boundaries_low[1] - 0.1).abs() < 1e-10);
        assert!((d.boundaries_high[1] - 0.9).abs() < 1e-10);
    }

    #[test]
    fn remap_scales_atom_positions() {
        let mut atoms = Atom::new();
        atoms.dt = 0.001;
        // Place atom at center of box
        atoms.push_test_atom(0, [0.5, 0.5, 0.5], 0.1, 1.0);
        // Place atom at 3/4 of box
        atoms.push_test_atom(1, [0.75, 0.5, 0.75], 0.1, 1.0);
        atoms.nlocal = 2;
        atoms.natoms = 2;

        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let neighbor = make_neighbor();
        let run_state = make_run_state(1000);

        let state = DeformState {
            axes: [
                None,
                None,
                Some(AxisDeform {
                    style: DeformStyle::Erate { rate: -1.0 },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
            ],
            remap: true,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        let d = app.get_resource_ref::<Domain>().unwrap();

        // Atom 0 at center should stay at center
        let center_z = (d.boundaries_low[2] + d.boundaries_high[2]) * 0.5;
        assert!(
            (a.pos[0][2] - center_z).abs() < 1e-12,
            "Center atom should stay at center: {} vs {}",
            a.pos[0][2],
            center_z
        );

        // x positions should be unchanged (no x deform)
        assert!((a.pos[0][0] - 0.5).abs() < 1e-12);
        assert!((a.pos[1][0] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn no_remap_leaves_atoms_in_place() {
        let mut atoms = Atom::new();
        atoms.dt = 0.001;
        atoms.push_test_atom(0, [0.5, 0.5, 0.5], 0.1, 1.0);
        atoms.nlocal = 1;
        atoms.natoms = 1;

        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let neighbor = make_neighbor();
        let run_state = make_run_state(1000);

        let state = DeformState {
            axes: [
                None,
                None,
                Some(AxisDeform {
                    style: DeformStyle::Erate { rate: -1.0 },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
            ],
            remap: false,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        // Position should be unchanged
        assert!((a.pos[0][2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn multiple_axes_deform_independently() {
        let mut atoms = make_atoms(0);
        atoms.dt = 0.01;
        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let neighbor = make_neighbor();
        let run_state = make_run_state(100);

        let state = DeformState {
            axes: [
                Some(AxisDeform {
                    style: DeformStyle::Vel { velocity: 1.0 },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
                None,
                Some(AxisDeform {
                    style: DeformStyle::Erate { rate: -0.5 },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
            ],
            remap: true,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let d = app.get_resource_ref::<Domain>().unwrap();

        // X: vel=1.0, dt=0.01, step=1 => delta=0.01, new_l=1.01
        assert!((d.size[0] - 1.01).abs() < 1e-12);

        // Y: unchanged
        assert!((d.size[1] - 1.0).abs() < 1e-12);

        // Z: erate=-0.5, dt=0.01, step=1 => scale = 1 + (-0.5)*0.01*1 = 0.995
        assert!((d.size[2] - 0.995).abs() < 1e-12);
    }

    #[test]
    fn volume_updates_correctly() {
        let mut atoms = make_atoms(0);
        atoms.dt = 0.001;
        let domain = make_domain([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
        let neighbor = make_neighbor();
        let run_state = make_run_state(1000);

        let state = DeformState {
            axes: [
                None,
                None,
                Some(AxisDeform {
                    style: DeformStyle::Erate { rate: -1.0 },
                    lo_0: 0.0,
                    hi_0: 4.0,
                }),
            ],
            remap: true,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let d = app.get_resource_ref::<Domain>().unwrap();
        let expected_vol = d.size[0] * d.size[1] * d.size[2];
        assert!((d.volume - expected_vol).abs() < 1e-10);
    }

    #[test]
    fn parse_erate_config() {
        let def = AxisDeformDef {
            style: "erate".to_string(),
            rate: Some(-0.001),
            velocity: None,
            lo: None,
            hi: None,
        };
        let axis = parse_axis_def(&Some(def), "z").unwrap();
        match axis.style {
            DeformStyle::Erate { rate } => assert!((rate - (-0.001)).abs() < 1e-15),
            _ => panic!("Expected Erate style"),
        }
    }

    #[test]
    fn parse_vel_config() {
        let def = AxisDeformDef {
            style: "vel".to_string(),
            rate: None,
            velocity: Some(0.5),
            lo: None,
            hi: None,
        };
        let axis = parse_axis_def(&Some(def), "x").unwrap();
        match axis.style {
            DeformStyle::Vel { velocity } => assert!((velocity - 0.5).abs() < 1e-15),
            _ => panic!("Expected Vel style"),
        }
    }

    #[test]
    fn parse_final_config() {
        let def = AxisDeformDef {
            style: "final".to_string(),
            rate: None,
            velocity: None,
            lo: Some(0.1),
            hi: Some(0.9),
        };
        let axis = parse_axis_def(&Some(def), "y").unwrap();
        match axis.style {
            DeformStyle::Final {
                target_lo,
                target_hi,
            } => {
                assert!((target_lo - 0.1).abs() < 1e-15);
                assert!((target_hi - 0.9).abs() < 1e-15);
            }
            _ => panic!("Expected Final style"),
        }
    }

    #[test]
    fn none_axis_returns_none() {
        assert!(parse_axis_def(&None, "x").is_none());
    }

    #[test]
    #[should_panic(expected = "requires 'rate' field")]
    fn erate_missing_rate_panics() {
        let def = AxisDeformDef {
            style: "erate".to_string(),
            rate: None,
            velocity: None,
            lo: None,
            hi: None,
        };
        parse_axis_def(&Some(def), "z");
    }

    #[test]
    #[should_panic(expected = "unknown style")]
    fn unknown_style_panics() {
        let def = AxisDeformDef {
            style: "wiggle".to_string(),
            rate: None,
            velocity: None,
            lo: None,
            hi: None,
        };
        parse_axis_def(&Some(def), "x");
    }
}

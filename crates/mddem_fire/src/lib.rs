//! FIRE (Fast Inertial Relaxation Engine) energy minimization.
//!
//! This crate implements the FIRE algorithm described in:
//!
//! > Bitzek, E., Koskinen, P., Gähler, F., Moseler, M., & Derlet, P. (2006).
//! > *Structural Relaxation Made Simple.* Physical Review Letters, 97(17), 170201.
//! > <https://doi.org/10.1103/PhysRevLett.97.170201>
//!
//! FIRE is a damped-dynamics minimizer that uses velocity feedback to accelerate
//! convergence toward a local energy minimum. It works by:
//!
//! 1. **Velocity Verlet integration** — positions and velocities are advanced
//!    using standard Verlet with an adaptive timestep.
//! 2. **Power check** — the instantaneous power `P = F · V` determines whether
//!    the system is moving downhill (`P > 0`) or uphill (`P < 0`).
//! 3. **Velocity mixing** — when `P > 0`, velocities are steered toward the
//!    force direction: `V ← (1 − α) V + α |V| (F / |F|)`.
//! 4. **Adaptive timestep** — after enough consecutive downhill steps, the
//!    timestep grows (up to `dt_max`); when `P < 0`, velocities are zeroed,
//!    the timestep shrinks, and the mixing parameter α is reset.
//!
//! Minimization converges when the maximum per-atom force magnitude falls below
//! the force tolerance `ftol`.
//!
//! # Usage
//!
//! Provides [`FireMinPlugin`] which replaces the standard Velocity Verlet
//! integrator with an adaptive-timestep FIRE minimizer. Configure via the
//! `[fire]` TOML section (see [`FireConfig`]).
//!
//! ```toml
//! [fire]
//! ftol = 1e-6          # force convergence tolerance
//! dt_max_factor = 10.0 # maximum dt = dt_max_factor × base dt
//! ```

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, Config};

// ── Config ──────────────────────────────────────────────────────────────────

/// TOML `[fire]` configuration section.
///
/// All fields have sensible defaults from the original FIRE paper. In most
/// cases only `ftol` needs to be tuned.
///
/// # Example
///
/// ```toml
/// [fire]
/// ftol = 1e-8
/// dt_max_factor = 5.0
/// n_delay = 10
/// ```
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct FireConfig {
    /// Force tolerance for convergence (force units).
    ///
    /// Minimization stops when the maximum per-atom force magnitude drops
    /// below this value. Smaller values yield tighter convergence but require
    /// more iterations.
    ///
    /// **Default:** `1e-6`
    #[serde(default = "default_ftol")]
    pub ftol: f64,

    /// Maximum timestep expressed as a multiple of the base `dt`.
    ///
    /// The adaptive timestep will never exceed `dt_max_factor × dt`. Larger
    /// values allow faster traversal of flat energy landscapes but may cause
    /// instability near steep features.
    ///
    /// **Default:** `10.0`
    #[serde(default = "default_dt_max_factor")]
    pub dt_max_factor: f64,

    /// Timestep growth factor applied when power is positive (`P > 0`).
    ///
    /// After `n_delay` consecutive positive-power steps, the timestep is
    /// multiplied by this factor (clamped to `dt_max`). Must be `> 1.0`.
    ///
    /// **Default:** `1.1`
    #[serde(default = "default_f_inc")]
    pub f_inc: f64,

    /// Timestep shrink factor applied when power is negative (`P < 0`).
    ///
    /// When the system moves uphill, the timestep is multiplied by this factor
    /// to pull back. Must be `< 1.0`.
    ///
    /// **Default:** `0.5`
    #[serde(default = "default_f_dec")]
    pub f_dec: f64,

    /// Initial velocity-mixing parameter α (dimensionless, 0–1).
    ///
    /// Controls how aggressively velocities are steered toward the force
    /// direction. `α = 0` means no mixing (pure MD), `α = 1` means velocities
    /// are set entirely along the force. The value decays toward zero via
    /// `f_alpha` as the system converges.
    ///
    /// **Default:** `0.1`
    #[serde(default = "default_alpha_start")]
    pub alpha_start: f64,

    /// Decay factor for α (dimensionless, 0–1).
    ///
    /// Each time the timestep grows (after `n_delay` positive-power steps),
    /// `α` is multiplied by this factor, gradually reducing the velocity
    /// mixing as the system approaches the minimum.
    ///
    /// **Default:** `0.99`
    #[serde(default = "default_f_alpha")]
    pub f_alpha: f64,

    /// Number of consecutive positive-power steps required before the
    /// timestep is allowed to grow.
    ///
    /// This delay prevents premature acceleration when the system is still
    /// oscillating. Larger values are more conservative.
    ///
    /// **Default:** `5`
    #[serde(default = "default_n_delay")]
    pub n_delay: usize,

    /// Whether to automatically advance to the next `[[run]]` stage when
    /// convergence is reached.
    ///
    /// Set to `false` if you want FIRE to continue running (as a no-op) after
    /// convergence rather than triggering a stage transition.
    ///
    /// **Default:** `true`
    #[serde(default = "default_true")]
    pub stop_on_converge: bool,
}

fn default_ftol() -> f64 { 1e-6 }
fn default_dt_max_factor() -> f64 { 10.0 }
fn default_f_inc() -> f64 { 1.1 }
fn default_f_dec() -> f64 { 0.5 }
fn default_alpha_start() -> f64 { 0.1 }
fn default_f_alpha() -> f64 { 0.99 }
fn default_n_delay() -> usize { 5 }
fn default_true() -> bool { true }

impl Default for FireConfig {
    fn default() -> Self {
        FireConfig {
            ftol: default_ftol(),
            dt_max_factor: default_dt_max_factor(),
            f_inc: default_f_inc(),
            f_dec: default_f_dec(),
            alpha_start: default_alpha_start(),
            f_alpha: default_f_alpha(),
            n_delay: default_n_delay(),
            stop_on_converge: true,
        }
    }
}

// ── Runtime state ───────────────────────────────────────────────────────────

/// Mutable runtime state for the FIRE minimizer.
///
/// Created automatically by [`FireMinPlugin::build`] from the base timestep
/// `dt` and [`FireConfig`]. Users typically inspect this (e.g. via
/// `Res<FireState>`) to check [`converged`](Self::converged) or read the
/// current [`iteration`](Self::iteration) count.
pub struct FireState {
    /// Current velocity-mixing parameter α (dimensionless, 0–1).
    ///
    /// Starts at [`FireConfig::alpha_start`] and decays by [`FireConfig::f_alpha`]
    /// each time the timestep grows. Reset to `alpha_start` on negative-power steps.
    pub alpha: f64,

    /// Current adaptive timestep used by FIRE integration (time units).
    ///
    /// Grows by [`FireConfig::f_inc`] after enough positive-power steps and
    /// shrinks by [`FireConfig::f_dec`] on negative-power steps.
    pub dt_fire: f64,

    /// Upper bound on `dt_fire` (time units), equal to `dt × dt_max_factor`.
    pub dt_max: f64,

    /// Number of consecutive steps with positive power (`P = F · V > 0`).
    ///
    /// Once this exceeds [`FireConfig::n_delay`], the timestep is allowed to
    /// grow and α decays. Reset to zero whenever `P ≤ 0`.
    pub steps_since_negative: usize,

    /// `true` once the maximum per-atom force magnitude drops below
    /// [`FireConfig::ftol`]. After convergence, FIRE integration becomes a no-op.
    pub converged: bool,

    /// Total number of FIRE iterations (final-integration calls) completed.
    pub iteration: usize,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Plugin providing FIRE energy minimization.
///
/// Implements the algorithm from Bitzek *et al.*, Phys. Rev. Lett. **97**, 170201 (2006).
///
/// When used alone (no `stage`), replaces Velocity Verlet entirely.
/// When `stage` is set, the FIRE systems only run during that `[[run]]` stage,
/// so it can coexist with [`VelocityVerletPlugin`] for multi-stage workflows
/// (e.g. minimize → dynamics).
///
/// # Examples
///
/// Single-stage (replaces Verlet):
/// ```rust,ignore
/// app.add_plugins(FireMinPlugin::new());
/// ```
///
/// Multi-stage with Verlet:
/// ```rust,ignore
/// app.add_plugins(GranularDefaultPlugins)             // includes Verlet
///     .add_plugins(FireMinPlugin::for_stage("minimize")); // FIRE only in "minimize"
/// ```
pub struct FireMinPlugin {
    /// If set, FIRE systems only run during this `[[run]]` stage name.
    pub stage: Option<String>,
}

impl FireMinPlugin {
    /// Create a FIRE plugin that runs in all stages (replaces Verlet).
    pub fn new() -> Self {
        Self { stage: None }
    }

    /// Create a FIRE plugin that only runs during the named `[[run]]` stage.
    pub fn for_stage(name: &str) -> Self {
        Self { stage: Some(name.to_string()) }
    }
}

impl Plugin for FireMinPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[fire]
# ftol = 1e-6
# dt_max_factor = 10.0
# f_inc = 1.1
# f_dec = 0.5
# alpha_start = 0.1
# f_alpha = 0.99
# n_delay = 5
# stop_on_converge = true"#,
        )
    }

    fn build(&self, app: &mut App) {
        let fire_config = Config::load::<FireConfig>(app, "fire");

        let atoms = app.get_resource_ref::<Atom>().expect("Atom resource must exist for FIRE");
        let dt = atoms.dt;
        drop(atoms);

        let state = FireState {
            alpha: fire_config.alpha_start,
            dt_fire: dt,
            dt_max: dt * fire_config.dt_max_factor,
            steps_since_negative: 0,
            converged: false,
            iteration: 0,
        };

        app.add_resource(state);

        if let Some(ref stage_name) = self.stage {
            app.add_update_system(
                fire_initial_integration.run_if(in_stage(stage_name)),
                ScheduleSet::InitialIntegration,
            );
            app.add_update_system(
                fire_final_integration.run_if(in_stage(stage_name)),
                ScheduleSet::FinalIntegration,
            );
        } else {
            app.add_update_system(
                fire_initial_integration,
                ScheduleSet::InitialIntegration,
            );
            app.add_update_system(
                fire_final_integration,
                ScheduleSet::FinalIntegration,
            );
        }
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// FIRE initial integration (Velocity Verlet first half-step).
///
/// Applies a half-step velocity kick (`v += 0.5 × dt × F/m`) followed by a
/// full-step position update (`x += v × dt`) using the adaptive FIRE timestep.
/// Skipped entirely once [`FireState::converged`] is `true`.
pub fn fire_initial_integration(
    mut atoms: ResMut<Atom>,
    fire: Res<FireState>,
) {
    if fire.converged {
        return;
    }

    let dt = fire.dt_fire;
    let nlocal = atoms.nlocal as usize;

    let inv_mass_ptr = atoms.inv_mass.as_ptr();
    let force_ptr = atoms.force.as_ptr();
    let vel_ptr = atoms.vel.as_mut_ptr();
    let pos_ptr = atoms.pos.as_mut_ptr();

    for i in 0..nlocal {
        unsafe {
            let half_dt_over_m = 0.5 * dt * *inv_mass_ptr.add(i);
            let f = &*force_ptr.add(i);
            let v = &mut *vel_ptr.add(i);
            v[0] += half_dt_over_m * f[0];
            v[1] += half_dt_over_m * f[1];
            v[2] += half_dt_over_m * f[2];
            let p = &mut *pos_ptr.add(i);
            p[0] += v[0] * dt;
            p[1] += v[1] * dt;
            p[2] += v[2] * dt;
        }
    }
}

/// FIRE final integration (Velocity Verlet second half-step + FIRE logic).
///
/// This system performs three steps after forces have been computed:
///
/// 1. **Second half-step velocity kick** — `v += 0.5 × dt × F/m`.
/// 2. **Convergence check** — if max per-atom `|F|` < `ftol`, mark converged.
/// 3. **FIRE velocity mixing and adaptive timestep** — steer velocities toward
///    the force direction and adjust `dt` based on whether the system is moving
///    downhill (positive power) or uphill (negative power).
///
/// Skipped entirely once [`FireState::converged`] is `true`.
pub fn fire_final_integration(
    mut atoms: ResMut<Atom>,
    mut fire: ResMut<FireState>,
    fire_config: Res<FireConfig>,
    mut scheduler: ResMut<SchedulerManager>,
) {
    if fire.converged {
        return;
    }

    let dt = fire.dt_fire;
    let nlocal = atoms.nlocal as usize;

    // Second half-step velocity kick
    let inv_mass_ptr = atoms.inv_mass.as_ptr();
    let force_ptr = atoms.force.as_ptr();
    let vel_ptr = atoms.vel.as_mut_ptr();

    for i in 0..nlocal {
        unsafe {
            let half_dt_over_m = 0.5 * dt * *inv_mass_ptr.add(i);
            let f = &*force_ptr.add(i);
            let v = &mut *vel_ptr.add(i);
            v[0] += half_dt_over_m * f[0];
            v[1] += half_dt_over_m * f[1];
            v[2] += half_dt_over_m * f[2];
        }
    }

    // Compute power P = F · V, magnitudes |F|, |V|, and max |F_i|
    let mut power = 0.0f64;
    let mut f_sq_sum = 0.0f64;
    let mut v_sq_sum = 0.0f64;
    let mut max_f_atom = 0.0f64;

    for i in 0..nlocal {
        let fx = atoms.force[i][0];
        let fy = atoms.force[i][1];
        let fz = atoms.force[i][2];
        let vx = atoms.vel[i][0];
        let vy = atoms.vel[i][1];
        let vz = atoms.vel[i][2];

        power += fx * vx + fy * vy + fz * vz;
        f_sq_sum += fx * fx + fy * fy + fz * fz;
        v_sq_sum += vx * vx + vy * vy + vz * vz;

        let f_mag = (fx * fx + fy * fy + fz * fz).sqrt();
        if f_mag > max_f_atom {
            max_f_atom = f_mag;
        }
    }

    fire.iteration += 1;

    // Check convergence
    if max_f_atom < fire_config.ftol {
        fire.converged = true;
        if fire_config.stop_on_converge {
            scheduler.advance_requested = true;
        }
        return;
    }

    // ── FIRE velocity mixing ────────────────────────────────────────────
    // Steer velocities toward the force direction while preserving speed:
    //   V ← (1 − α) V  +  α |V| (F / |F|)
    // The first term retains most of the current velocity; the second term
    // adds a component along the net force, weighted by α.
    let f_mag_total = f_sq_sum.sqrt();
    let v_mag_total = v_sq_sum.sqrt();

    if f_mag_total > 1e-30 {
        let alpha = fire.alpha;
        let scale = alpha * v_mag_total / f_mag_total;

        for i in 0..nlocal {
            atoms.vel[i][0] = (1.0 - alpha) * atoms.vel[i][0] + scale * atoms.force[i][0];
            atoms.vel[i][1] = (1.0 - alpha) * atoms.vel[i][1] + scale * atoms.force[i][1];
            atoms.vel[i][2] = (1.0 - alpha) * atoms.vel[i][2] + scale * atoms.force[i][2];
        }
    }

    // ── FIRE adaptive timestep (state machine) ───────────────────────────
    if power > 0.0 {
        // System is moving downhill — accumulate positive-power steps.
        fire.steps_since_negative += 1;
        if fire.steps_since_negative > fire_config.n_delay {
            // Enough consecutive downhill steps: accelerate by growing dt
            // and reducing the mixing parameter α toward zero.
            fire.dt_fire = (fire.dt_fire * fire_config.f_inc).min(fire.dt_max);
            fire.alpha *= fire_config.f_alpha;
        }
    } else {
        // System is moving uphill (P ≤ 0) — emergency brake:
        //   • shrink the timestep to reduce overshoot,
        //   • reset α to its initial (aggressive) value,
        //   • zero all velocities so the system restarts from rest.
        fire.dt_fire *= fire_config.f_dec;
        fire.alpha = fire_config.alpha_start;
        fire.steps_since_negative = 0;
        for i in 0..nlocal {
            atoms.vel[i] = [0.0; 3];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::Atom;

    fn make_fire_state(dt: f64) -> FireState {
        let config = FireConfig::default();
        FireState {
            alpha: config.alpha_start,
            dt_fire: dt,
            dt_max: dt * config.dt_max_factor,
            steps_since_negative: 0,
            converged: false,
            iteration: 0,
        }
    }

    #[test]
    fn fire_single_particle_spring_converges() {
        // Particle at x=0.1 with spring force F = -k*x toward origin
        // After enough FIRE iterations, should converge to x≈0
        let mut atom = Atom::new();
        atom.dt = 0.001;
        atom.push_test_atom(0, [0.1, 0.0, 0.0], 0.001, 1.0);
        atom.nlocal = 1;
        atom.natoms = 1;

        let fire_state = make_fire_state(0.001);
        let fire_config = FireConfig {
            ftol: 1e-4,
            ..FireConfig::default()
        };
        let scheduler = SchedulerManager::default();

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(fire_state);
        app.add_resource(fire_config);
        app.add_resource(scheduler);

        // Simulate spring force manually: apply force, run FIRE, repeat
        let k = 100.0;
        for _ in 0..1000 {
            {
                let atom = app.get_resource_ref::<Atom>().unwrap();
                // Already converged?
                let fire = app.get_resource_ref::<FireState>().unwrap();
                if fire.converged {
                    break;
                }
                drop(fire);
                drop(atom);
            }

            // Apply spring force
            {
                let atom_cell = app.get_mut_resource(std::any::TypeId::of::<Atom>()).unwrap();
                let mut borrow = atom_cell.borrow_mut();
                let atom = borrow.downcast_mut::<Atom>().unwrap();
                atom.force[0] = [-k * atom.pos[0][0], -k * atom.pos[0][1], -k * atom.pos[0][2]];
            }

            // Run FIRE integration
            app.add_update_system(fire_initial_integration, ScheduleSet::InitialIntegration);
            app.add_update_system(fire_final_integration, ScheduleSet::FinalIntegration);
            app.organize_systems();
            app.run();
        }

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let fire = app.get_resource_ref::<FireState>().unwrap();
        // Should either converge or get close to origin
        assert!(
            atom.pos[0][0].abs() < 0.01 || fire.converged,
            "particle should move toward origin, pos={}, converged={}",
            atom.pos[0][0],
            fire.converged
        );
    }

    #[test]
    fn fire_velocities_zeroed_when_power_negative() {
        let mut atom = Atom::new();
        atom.dt = 0.001;
        atom.push_test_atom(0, [0.0, 0.0, 0.0], 0.001, 1.0);
        atom.vel[0] = [1.0, 0.0, 0.0];
        // Force opposes velocity → P < 0
        atom.force[0] = [-10.0, 0.0, 0.0];
        atom.nlocal = 1;
        atom.natoms = 1;

        let fire_state = make_fire_state(0.001);
        let fire_config = FireConfig::default();
        let scheduler = SchedulerManager::default();

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(fire_state);
        app.add_resource(fire_config);
        app.add_resource(scheduler);
        app.add_update_system(fire_final_integration, ScheduleSet::FinalIntegration);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // After half-step kick, power F·V should still be negative
        // Velocities should be zeroed
        assert!(
            atom.vel[0][0].abs() < 1e-15,
            "velocity should be zeroed when P < 0, got {}",
            atom.vel[0][0]
        );
    }

    #[test]
    fn fire_dt_increases_after_delay() {
        let mut atom = Atom::new();
        atom.dt = 0.001;
        atom.push_test_atom(0, [0.0, 0.0, 0.0], 0.001, 1.0);
        atom.vel[0] = [1.0, 0.0, 0.0];
        // Force aligned with velocity → P > 0
        atom.force[0] = [10.0, 0.0, 0.0];
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut fire_state = make_fire_state(0.001);
        fire_state.steps_since_negative = 10; // past n_delay
        let initial_dt = fire_state.dt_fire;

        let fire_config = FireConfig::default();
        let scheduler = SchedulerManager::default();

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(fire_state);
        app.add_resource(fire_config);
        app.add_resource(scheduler);
        app.add_update_system(fire_final_integration, ScheduleSet::FinalIntegration);
        app.organize_systems();
        app.run();

        let fire = app.get_resource_ref::<FireState>().unwrap();
        assert!(
            fire.dt_fire > initial_dt,
            "dt should increase: initial={}, current={}",
            initial_dt,
            fire.dt_fire
        );
    }

    #[test]
    fn fire_convergence_sets_advance() {
        let mut atom = Atom::new();
        atom.dt = 0.001;
        atom.push_test_atom(0, [0.0, 0.0, 0.0], 0.001, 1.0);
        // Very small force below ftol
        atom.force[0] = [1e-8, 0.0, 0.0];
        atom.nlocal = 1;
        atom.natoms = 1;

        let fire_state = make_fire_state(0.001);
        let fire_config = FireConfig {
            ftol: 1e-6,
            ..FireConfig::default()
        };
        let scheduler = SchedulerManager::default();

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(fire_state);
        app.add_resource(fire_config);
        app.add_resource(scheduler);
        app.add_update_system(fire_final_integration, ScheduleSet::FinalIntegration);
        app.organize_systems();
        app.run();

        let fire = app.get_resource_ref::<FireState>().unwrap();
        let sched = app.get_resource_ref::<SchedulerManager>().unwrap();
        assert!(fire.converged, "FIRE should converge");
        assert!(sched.advance_requested, "should request stage advance");
    }

    #[test]
    fn fire_config_deserialization() {
        let toml_str = r#"
ftol = 1e-8
dt_max_factor = 5.0
n_delay = 10
"#;
        let config: FireConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.ftol, 1e-8);
        assert_eq!(config.dt_max_factor, 5.0);
        assert_eq!(config.n_delay, 10);
        // Defaults
        assert_eq!(config.f_inc, 1.1);
        assert_eq!(config.alpha_start, 0.1);
    }
}

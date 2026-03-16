//! FIRE (Fast Inertial Relaxation Engine) energy minimization.
//!
//! Provides [`FireMinPlugin`] which replaces the standard Velocity Verlet
//! integrator with an adaptive-timestep FIRE minimizer. Converges when the
//! maximum per-atom force magnitude falls below `ftol`.

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, Config};

// ── Config ──────────────────────────────────────────────────────────────────

/// TOML `[fire]` configuration section.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct FireConfig {
    /// Force tolerance for convergence (max per-atom force magnitude).
    #[serde(default = "default_ftol")]
    pub ftol: f64,
    /// Maximum timestep as a multiple of the base dt.
    #[serde(default = "default_dt_max_factor")]
    pub dt_max_factor: f64,
    /// Factor to increase dt when power is positive.
    #[serde(default = "default_f_inc")]
    pub f_inc: f64,
    /// Factor to decrease dt when power is negative.
    #[serde(default = "default_f_dec")]
    pub f_dec: f64,
    /// Initial mixing parameter.
    #[serde(default = "default_alpha_start")]
    pub alpha_start: f64,
    /// Factor to decrease alpha when power is positive.
    #[serde(default = "default_f_alpha")]
    pub f_alpha: f64,
    /// Number of positive-power steps before increasing dt.
    #[serde(default = "default_n_delay")]
    pub n_delay: usize,
    /// Whether to advance to next stage on convergence.
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

/// FIRE algorithm runtime state.
pub struct FireState {
    /// Current mixing parameter alpha.
    pub alpha: f64,
    /// Current FIRE timestep.
    pub dt_fire: f64,
    /// Maximum allowed FIRE timestep.
    pub dt_max: f64,
    /// Consecutive positive-power steps.
    pub steps_since_negative: usize,
    /// Whether minimization has converged.
    pub converged: bool,
    /// Iteration counter.
    pub iteration: usize,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// FIRE minimization plugin.
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

/// FIRE initial integration: half-step velocity kick + position update.
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

/// FIRE final integration: half-step velocity kick + FIRE mixing + adaptive dt.
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

    // FIRE velocity mixing: V = (1-α)*V + α*|V|*(F/|F|)
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

    // FIRE adaptive timestep
    if power > 0.0 {
        fire.steps_since_negative += 1;
        if fire.steps_since_negative > fire_config.n_delay {
            fire.dt_fire = (fire.dt_fire * fire_config.f_inc).min(fire.dt_max);
            fire.alpha *= fire_config.f_alpha;
        }
    } else {
        // Power is negative: reset
        fire.dt_fire *= fire_config.f_dec;
        fire.alpha = fire_config.alpha_start;
        fire.steps_since_negative = 0;
        // Zero velocities
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
                let mut atom = app.get_resource_ref::<Atom>().unwrap();
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

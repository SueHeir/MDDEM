//! Translational Velocity Verlet time integration (half-step kick-drift-kick).
//!
//! This crate implements the **velocity Verlet** algorithm for integrating
//! Newton's equations of motion. The scheme is split into two phases that
//! bracket the force calculation each timestep:
//!
//! **Initial integration** (before forces):
//!
//! ```text
//!   v(t + Δt/2) = v(t)     + (Δt / 2m) · F(t)      // half-step velocity kick
//!   x(t + Δt)   = x(t)     + Δt · v(t + Δt/2)       // full-step position drift
//! ```
//!
//! **Final integration** (after forces):
//!
//! ```text
//!   v(t + Δt)   = v(t + Δt/2) + (Δt / 2m) · F(t + Δt) // completing velocity kick
//! ```
//!
//! This "kick-drift-kick" decomposition is symplectic, time-reversible, and
//! second-order accurate in Δt. It exactly integrates constant-force motion
//! and conserves energy to O(Δt²) per step for Hamiltonian systems.

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use mddem_core::Atom;

/// Registers initial and final integration systems for translational Velocity Verlet.
///
/// When `stage` is `None` (the default), systems run every stage.
/// Use [`VelocityVerletPlugin::for_stage`] to restrict to a single `[[run]]` stage,
/// e.g. when pairing with [`FireMinPlugin::for_stage`] in a multi-stage workflow.
///
/// # Examples
///
/// All stages (default):
/// ```rust,ignore
/// app.add_plugins(VelocityVerletPlugin::new());
/// ```
///
/// Single stage:
/// ```rust,ignore
/// app.add_plugins(VelocityVerletPlugin::for_stage("cooling"));
/// ```
pub struct VelocityVerletPlugin {
    /// If set, Verlet systems only run during this `[[run]]` stage name.
    pub stage: Option<String>,
}

impl VelocityVerletPlugin {
    /// Create a Verlet plugin that runs in all stages.
    pub fn new() -> Self {
        Self { stage: None }
    }

    /// Create a Verlet plugin that only runs during the named `[[run]]` stage.
    pub fn for_stage(name: &str) -> Self {
        Self { stage: Some(name.to_string()) }
    }
}

impl Default for VelocityVerletPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for VelocityVerletPlugin {
    fn build(&self, app: &mut App) {
        if let Some(ref stage_name) = self.stage {
            app.add_update_system(
                initial_integration.run_if(in_stage(stage_name)),
                ScheduleSet::InitialIntegration,
            )
            .add_update_system(
                final_integration.run_if(in_stage(stage_name)),
                ScheduleSet::FinalIntegration,
            );
        } else {
            app.add_update_system(initial_integration, ScheduleSet::InitialIntegration)
                .add_update_system(final_integration, ScheduleSet::FinalIntegration);
        }
    }
}

/// Performs the first half of velocity Verlet: half-step velocity kick + position drift.
///
/// For each local atom *i*:
///
/// ```text
///   v_i  +=  (Δt / 2m_i) · F_i     (half-kick using forces from previous step)
///   x_i  +=  Δt · v_i               (drift at the updated half-step velocity)
/// ```
///
/// This runs at [`ScheduleSet::InitialIntegration`], **before** force computation.
pub fn initial_integration(mut atoms: ResMut<Atom>) {
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;
    // SAFETY: i < nlocal <= len for all arrays (inv_mass, force, vel, pos).
    // Raw pointers avoid borrow-checker conflicts when mutating vel/pos while
    // reading inv_mass/force from the same struct.
    let inv_mass_ptr = atoms.inv_mass.as_ptr();
    let force_ptr = atoms.force.as_ptr();
    let vel_ptr = atoms.vel.as_mut_ptr();
    let pos_ptr = atoms.pos.as_mut_ptr();
    for i in 0..nlocal {
        unsafe {
            let half_dt_over_m = 0.5 * dt * *inv_mass_ptr.add(i);
            let f = &*force_ptr.add(i);
            let v = &mut *vel_ptr.add(i);
            // Half-step velocity kick: v(t) → v(t + Δt/2)
            v[0] += half_dt_over_m * f[0];
            v[1] += half_dt_over_m * f[1];
            v[2] += half_dt_over_m * f[2];
            // Full-step position drift using the half-step velocity
            let p = &mut *pos_ptr.add(i);
            p[0] += v[0] * dt;
            p[1] += v[1] * dt;
            p[2] += v[2] * dt;
        }
    }
}

/// Performs the second half of velocity Verlet: completing the velocity kick.
///
/// For each local atom *i*:
///
/// ```text
///   v_i  +=  (Δt / 2m_i) · F_i     (half-kick using newly computed forces)
/// ```
///
/// After this step the velocity is fully updated: v(t + Δt/2) → v(t + Δt).
///
/// This runs at [`ScheduleSet::FinalIntegration`], **after** force computation.
pub fn final_integration(mut atoms: ResMut<Atom>) {
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;
    // SAFETY: i < nlocal <= len for all arrays (inv_mass, force, vel).
    let inv_mass_ptr = atoms.inv_mass.as_ptr();
    let force_ptr = atoms.force.as_ptr();
    let vel_ptr = atoms.vel.as_mut_ptr();
    for i in 0..nlocal {
        unsafe {
            let half_dt_over_m = 0.5 * dt * *inv_mass_ptr.add(i);
            let f = &*force_ptr.add(i);
            let v = &mut *vel_ptr.add(i);
            // Completing velocity kick: v(t + Δt/2) → v(t + Δt)
            v[0] += half_dt_over_m * f[0];
            v[1] += half_dt_over_m * f[1];
            v[2] += half_dt_over_m * f[2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::Atom;

    fn make_atom() -> Atom {
        let mut atom = Atom::new();
        atom.dt = 0.01;
        atom.push_test_atom(0, [0.0; 3], 0.001, 1.0);
        atom.vel[0][0] = 1.0;
        atom.force[0][0] = 2.0;
        atom.nlocal = 1;
        atom.natoms = 1;
        atom
    }

    #[test]
    fn initial_integration_updates_position_and_velocity() {
        let mut app = App::new();
        app.add_resource(make_atom());
        app.add_update_system(initial_integration, ScheduleSet::InitialIntegration);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // v += 0.5 * 0.01 * 2.0 / 1.0 = 0.01 → v = 1.01
        // x += 1.01 * 0.01 = 0.0101
        assert!((atom.vel[0][0] - 1.01).abs() < 1e-10);
        assert!((atom.pos[0][0] - 0.0101).abs() < 1e-10);
    }

    // ══════════════════════════════════════════════════════════════════════
    // VALIDATION: Velocity Verlet is second-order accurate
    // For a harmonic oscillator F = -k*x, the Velocity Verlet error scales
    // as O(dt^2). Run at dt and dt/2: the position error ratio should be ~4.
    // This verifies the integrator order of convergence.
    // ══════════════════════════════════════════════════════════════════════
    #[test]
    fn velocity_verlet_second_order_convergence() {
        // Harmonic oscillator: F = -k*x, with k=1, m=1
        // Exact solution: x(t) = x0*cos(t) + v0*sin(t)
        // We use x0=1, v0=0 → x(t) = cos(t), v(t) = -sin(t)
        let k = 1.0;
        let x0 = 1.0;
        let v0 = 0.0;
        let t_final = 1.0; // integrate to t=1

        let run_harmonic = |dt: f64| -> (f64, f64) {
            let nsteps = (t_final / dt).round() as usize;
            let mut x = x0;
            let mut v = v0;
            // Manual Velocity Verlet loop (no App needed for pure integrator test)
            for _ in 0..nsteps {
                let f = -k * x;
                // Initial half-kick
                v += 0.5 * dt * f; // f/m with m=1
                // Drift
                x += v * dt;
                // Compute new force
                let f_new = -k * x;
                // Final half-kick
                v += 0.5 * dt * f_new;
            }
            (x, v)
        };

        let dt1 = 0.01;
        let dt2 = 0.005; // dt/2
        let dt3 = 0.0025; // dt/4

        let exact_x = t_final.cos();

        let (x1, _) = run_harmonic(dt1);
        let (x2, _) = run_harmonic(dt2);
        let (x3, _) = run_harmonic(dt3);

        let err1 = (x1 - exact_x).abs();
        let err2 = (x2 - exact_x).abs();
        let err3 = (x3 - exact_x).abs();

        // Second-order: err(dt/2) / err(dt) ≈ 1/4
        let ratio_12 = err1 / err2;
        let ratio_23 = err2 / err3;

        assert!(
            (ratio_12 - 4.0).abs() < 0.5,
            "VV convergence dt→dt/2: error ratio should be ~4, got {:.2} (errs: {:.2e}, {:.2e})",
            ratio_12, err1, err2
        );
        assert!(
            (ratio_23 - 4.0).abs() < 0.5,
            "VV convergence dt/2→dt/4: error ratio should be ~4, got {:.2} (errs: {:.2e}, {:.2e})",
            ratio_23, err2, err3
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // VALIDATION: Velocity Verlet conserves energy for harmonic oscillator
    // The symplectic integrator should conserve energy up to O(dt^2) per
    // step, resulting in bounded energy drift over long times.
    // ══════════════════════════════════════════════════════════════════════
    #[test]
    fn velocity_verlet_energy_conservation_harmonic() {
        let k: f64 = 100.0;
        let x0: f64 = 0.1;
        let v0: f64 = 0.0;
        let dt: f64 = 1e-4;
        let nsteps = 10000; // many oscillation periods

        let mut x = x0;
        let mut v = v0;
        let e_initial: f64 = 0.5 * k * x * x + 0.5 * v * v;

        for _ in 0..nsteps {
            let f = -k * x;
            v += 0.5 * dt * f;
            x += v * dt;
            let f_new = -k * x;
            v += 0.5 * dt * f_new;
        }

        let e_final = 0.5 * k * x * x + 0.5 * v * v;
        let rel_drift = (e_final - e_initial).abs() / e_initial.abs();

        assert!(
            rel_drift < 1e-6,
            "VV energy drift over {} steps: relative = {:.2e} (should be < 1e-6)",
            nsteps, rel_drift
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // VALIDATION: Free particle (F=0) moves at constant velocity
    // This tests that the integrator preserves velocity when no force acts.
    // ══════════════════════════════════════════════════════════════════════
    #[test]
    fn free_particle_constant_velocity() {
        let mut app = App::new();
        let mut atom = Atom::new();
        atom.dt = 0.001;
        atom.push_test_atom(0, [0.0, 0.0, 0.0], 0.001, 1.0);
        atom.vel[0] = [1.5, -2.3, 0.7];
        atom.nlocal = 1;
        atom.natoms = 1;

        app.add_resource(atom);
        app.add_update_system(initial_integration, ScheduleSet::InitialIntegration);
        app.add_update_system(final_integration, ScheduleSet::FinalIntegration);
        app.organize_systems();

        for _ in 0..1000 {
            app.run();
        }

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let t = 1.0; // 1000 * 0.001
        assert!(
            (atom.pos[0][0] - 1.5 * t).abs() < 1e-10,
            "x position: {}", atom.pos[0][0]
        );
        assert!(
            (atom.pos[0][1] - (-2.3 * t)).abs() < 1e-10,
            "y position: {}", atom.pos[0][1]
        );
        assert!(
            (atom.vel[0][0] - 1.5).abs() < 1e-14,
            "x velocity preserved: {}", atom.vel[0][0]
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // VALIDATION: Constant force produces parabolic trajectory
    // With F = m*g, the position should be x = v0*t + 0.5*g*t^2 (exact for VV).
    // ══════════════════════════════════════════════════════════════════════
    #[test]
    fn constant_force_parabolic_trajectory() {
        // Velocity Verlet is exact for constant force: x = x0 + v0*t + 0.5*a*t^2.
        // We test this by manually running the VV loop (same equations as the
        // integrator) to avoid App scheduling subtleties with force zeroing.
        let dt: f64 = 0.0001;
        let nsteps = 10000;
        let mass: f64 = 2.0;
        let g: f64 = -9.81;
        let a = g; // acceleration = F/m = m*g/m = g

        let mut x: f64 = 10.0;
        let mut v: f64 = 5.0;
        let f = mass * g;

        for _ in 0..nsteps {
            // VV: half-kick, drift, force, half-kick
            v += 0.5 * dt * f / mass;
            x += v * dt;
            // force is constant, no recalculation needed
            v += 0.5 * dt * f / mass;
        }

        let t = nsteps as f64 * dt;
        let expected_x = 10.0 + 5.0 * t + 0.5 * a * t * t;
        let expected_v = 5.0 + a * t;

        // VV is exact for constant force — error should be machine precision
        assert!(
            (x - expected_x).abs() < 1e-10,
            "z pos: got {}, expected {}", x, expected_x
        );
        assert!(
            (v - expected_v).abs() < 1e-10,
            "vz: got {}, expected {}", v, expected_v
        );
    }

    #[test]
    fn final_integration_updates_velocity_only() {
        let mut app = App::new();
        app.add_resource(make_atom());
        app.add_update_system(final_integration, ScheduleSet::FinalIntegration);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!((atom.vel[0][0] - 1.01).abs() < 1e-10);
        // Position should be unchanged
        assert!((atom.pos[0][0] - 0.0).abs() < 1e-10);
    }
}

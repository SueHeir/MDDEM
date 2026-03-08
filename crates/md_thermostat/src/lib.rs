use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, CommResource, Config};

// ── Config ──────────────────────────────────────────────────────────────────

fn default_temperature() -> f64 {
    0.85
}
fn default_coupling() -> f64 {
    1.0
}

#[derive(Deserialize, Clone)]
pub struct ThermostatConfig {
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_coupling")]
    pub coupling: f64,
}

impl Default for ThermostatConfig {
    fn default() -> Self {
        ThermostatConfig {
            temperature: 0.85,
            coupling: 1.0,
        }
    }
}

// ── Resource ────────────────────────────────────────────────────────────────

pub struct NoseHooverState {
    pub p_xi: f64,
    pub q_mass: f64,
    pub target_temp: f64,
    pub ndof: f64,
}

impl Default for NoseHooverState {
    fn default() -> Self {
        NoseHooverState {
            p_xi: 0.0,
            q_mass: 1.0,
            target_temp: 0.85,
            ndof: 3.0,
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

pub struct NoseHooverPlugin;

impl Plugin for NoseHooverPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[thermostat]
temperature = 0.85   # target T* (reduced units)
coupling = 1.0       # relaxation time tau_T"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<ThermostatConfig>(app, "thermostat");

        app.add_resource(NoseHooverState::default())
            .add_setup_system(setup_nose_hoover, ScheduleSetupSet::PostSetup)
            .add_update_system(nh_pre_initial, ScheduleSet::PreInitialIntegration)
            .add_update_system(nh_post_final, ScheduleSet::PostFinalIntegration);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

pub fn setup_nose_hoover(
    config: Res<ThermostatConfig>,
    atoms: Res<Atom>,
    comm: Res<CommResource>,
    mut nh: ResMut<NoseHooverState>,
    scheduler_manager: Res<SchedulerManager>,
) {
    if scheduler_manager.index != 0 {
        return;
    }

    let n = comm.all_reduce_sum_f64(atoms.nlocal as f64);
    let ndof = 3.0 * n - 3.0; // subtract COM degrees of freedom
    let tau = config.coupling;
    let t_target = config.temperature;
    let q_mass = ndof * t_target * tau * tau;

    nh.ndof = ndof;
    nh.target_temp = t_target;
    nh.q_mass = q_mass;
    nh.p_xi = 0.0;

    if comm.rank() == 0 {
        println!(
            "Nose-Hoover: T*={}, tau={}, ndof={}, Q={}",
            t_target, tau, ndof, q_mass
        );
    }
}

fn compute_ke(atoms: &Atom) -> f64 {
    let nlocal = atoms.nlocal as usize;
    let mut ke = 0.0;
    for i in 0..nlocal {
        let vx = atoms.vel_x[i];
        let vy = atoms.vel_y[i];
        let vz = atoms.vel_z[i];
        ke += atoms.mass[i] * (vx * vx + vy * vy + vz * vz);
    }
    0.5 * ke
}

/// Pre-initial integration: half-step NH thermostat
/// 1. Compute KE (global reduction for MPI)
/// 2. Update p_xi by half step
/// 3. Rescale velocities by exp(-dt/2 * p_xi/Q)
pub fn nh_pre_initial(mut atoms: ResMut<Atom>, mut nh: ResMut<NoseHooverState>, comm: Res<CommResource>) {
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;

    let ke_local = compute_ke(&atoms);
    let ke = comm.all_reduce_sum_f64(ke_local);
    nh.p_xi += (dt / 2.0) * (2.0 * ke - nh.ndof * nh.target_temp);

    let scale = (-dt / 2.0 * nh.p_xi / nh.q_mass).exp();
    for i in 0..nlocal {
        atoms.vel_x[i] *= scale;
        atoms.vel_y[i] *= scale;
        atoms.vel_z[i] *= scale;
    }
}

/// Post-final integration: second half-step NH thermostat
/// 1. Rescale velocities by exp(-dt/2 * p_xi/Q)
/// 2. Recompute KE (global reduction for MPI)
/// 3. Update p_xi by half step
pub fn nh_post_final(mut atoms: ResMut<Atom>, mut nh: ResMut<NoseHooverState>, comm: Res<CommResource>) {
    let dt = atoms.dt;
    let nlocal = atoms.nlocal as usize;

    let scale = (-dt / 2.0 * nh.p_xi / nh.q_mass).exp();
    for i in 0..nlocal {
        atoms.vel_x[i] *= scale;
        atoms.vel_y[i] *= scale;
        atoms.vel_z[i] *= scale;
    }

    let ke_local = compute_ke(&atoms);
    let ke = comm.all_reduce_sum_f64(ke_local);
    nh.p_xi += (dt / 2.0) * (2.0 * ke - nh.ndof * nh.target_temp);
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    fn push_atom(atom: &mut Atom, tag: u32) {
        use nalgebra::{Quaternion, UnitQuaternion};
        atom.tag.push(tag);
        atom.atom_type.push(0);
        atom.origin_index.push(0);
        atom.pos_x.push(0.0);
        atom.pos_y.push(0.0);
        atom.pos_z.push(0.0);
        atom.vel_x.push(0.0);
        atom.vel_y.push(0.0);
        atom.vel_z.push(0.0);
        atom.force_x.push(0.0);
        atom.force_y.push(0.0);
        atom.force_z.push(0.0);
        atom.torque_x.push(0.0);
        atom.torque_y.push(0.0);
        atom.torque_z.push(0.0);
        atom.omega_x.push(0.0);
        atom.omega_y.push(0.0);
        atom.omega_z.push(0.0);
        atom.ang_mom_x.push(0.0);
        atom.ang_mom_y.push(0.0);
        atom.ang_mom_z.push(0.0);
        atom.mass.push(1.0);
        atom.skin.push(0.5);
        atom.is_ghost.push(false);
        atom.has_ghost.push(false);
        atom.is_collision.push(false);
        atom.quaterion
            .push(UnitQuaternion::from_quaternion(Quaternion::identity()));
    }

    fn make_nh_app(n: usize, velocities: &[(f64, f64, f64)], t_target: f64, tau: f64) -> App {
        let mut app = App::new();

        let ndof = 3.0 * n as f64 - 3.0;
        let q_mass = ndof * t_target * tau * tau;

        let config = ThermostatConfig {
            temperature: t_target,
            coupling: tau,
        };
        let nh = NoseHooverState {
            p_xi: 0.0,
            q_mass,
            target_temp: t_target,
            ndof,
        };

        let mut atom = Atom::new();
        atom.dt = 0.001;
        for i in 0..n {
            push_atom(&mut atom, i as u32);
            atom.vel_x[i] = velocities[i].0;
            atom.vel_y[i] = velocities[i].1;
            atom.vel_z[i] = velocities[i].2;
        }
        atom.nlocal = n as u32;
        atom.natoms = n as u64;

        app.add_resource(config);
        app.add_resource(nh);
        app.add_resource(atom);
        app.add_resource(CommResource(Box::new(
            mddem_core::SingleProcessComm::new(),
        )));
        app.add_update_system(nh_pre_initial, ScheduleSet::PreInitialIntegration);
        app.add_update_system(nh_post_final, ScheduleSet::PostFinalIntegration);
        app.organize_systems();
        app
    }

    #[test]
    fn nh_temperature_control() {
        let n = 100;
        let vels: Vec<_> = (0..n).map(|_| (1.0, 1.0, 1.0)).collect();
        let ndof = 3.0 * n as f64 - 3.0;

        let mut app = make_nh_app(n, &vels, 1.0, 0.5);

        let ke_before = {
            let a = app.get_resource_ref::<Atom>().unwrap();
            compute_ke(&a)
        };

        for _ in 0..1000 {
            app.run();
        }

        let ke_after = {
            let a = app.get_resource_ref::<Atom>().unwrap();
            compute_ke(&a)
        };

        let t_before = 2.0 * ke_before / ndof;
        let t_after = 2.0 * ke_after / ndof;
        assert!(
            t_after < t_before,
            "Temperature should decrease: before={}, after={}",
            t_before,
            t_after
        );
    }

    #[test]
    fn nh_conserved_quantity_stable() {
        let n = 50;
        let vels: Vec<_> = (0..n)
            .map(|i| {
                (
                    ((i as f64) * 0.1).sin(),
                    ((i as f64) * 0.2).cos(),
                    ((i as f64) * 0.3).sin(),
                )
            })
            .collect();

        let mut app = make_nh_app(n, &vels, 1.0, 1.0);

        for _ in 0..500 {
            app.run();
        }

        let a = app.get_resource_ref::<Atom>().unwrap();
        let nh_state = app.get_resource_ref::<NoseHooverState>().unwrap();
        let ke = compute_ke(&a);
        assert!(ke.is_finite(), "KE should be finite: {}", ke);
        assert!(
            nh_state.p_xi.is_finite(),
            "p_xi should be finite: {}",
            nh_state.p_xi
        );
    }
}

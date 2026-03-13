//! Thermostats: Nose-Hoover NVT (symmetric Liouville splitting) and Langevin (stochastic).

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use serde::Deserialize;

use mddem_core::{compute_ke, group_includes, Atom, CommResource, Config, GroupRegistry};

// ── Config ──────────────────────────────────────────────────────────────────

fn default_temperature() -> f64 {
    0.85
}
fn default_coupling() -> f64 {
    1.0
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[thermostat]` — Nose-Hoover thermostat settings.
pub struct ThermostatConfig {
    /// Target temperature (reduced units).
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Coupling time constant (reduced units).
    #[serde(default = "default_coupling")]
    pub coupling: f64,
    /// Optional group name — only thermostat atoms in this group.
    #[serde(default)]
    pub group: Option<String>,
}

impl Default for ThermostatConfig {
    fn default() -> Self {
        ThermostatConfig {
            temperature: 0.85,
            coupling: 1.0,
            group: None,
        }
    }
}

// ── Resource ────────────────────────────────────────────────────────────────

/// Internal state of the Nose-Hoover thermostat chain variable.
pub struct NoseHooverState {
    pub p_xi: f64,
    pub q_mass: f64,
    pub target_temp: f64,
    pub ndof: f64,
    pub group_name: Option<String>,
}

impl Default for NoseHooverState {
    fn default() -> Self {
        NoseHooverState {
            p_xi: 0.0,
            q_mass: 1.0,
            target_temp: 0.85,
            ndof: 3.0,
            group_name: None,
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Registers Nose-Hoover NVT thermostat integration systems.
pub struct NoseHooverPlugin;

impl Plugin for NoseHooverPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[thermostat]
temperature = 0.85   # target T* (reduced units)
coupling = 1.0       # relaxation time tau_T
# group = "mobile"   # optional: only thermostat this group"#,
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
    groups: Res<GroupRegistry>,
    mut nh: ResMut<NoseHooverState>,
    scheduler_manager: Res<SchedulerManager>,
) {
    if scheduler_manager.index != 0 {
        return;
    }

    if let Some(ref gname) = config.group {
        groups.validate_name(gname, "Nose-Hoover thermostat");
    }

    nh.group_name = config.group.clone();

    let n = if let Some(ref gname) = config.group {
        let group = groups.expect(gname);
        comm.all_reduce_sum_f64(group.count as f64)
    } else {
        comm.all_reduce_sum_f64(atoms.nlocal as f64)
    };

    let ndof = 3.0 * n - 3.0; // subtract COM degrees of freedom
    let tau = config.coupling;
    let t_target = config.temperature;
    let q_mass = ndof * t_target * tau * tau;

    nh.ndof = ndof;
    nh.target_temp = t_target;
    nh.q_mass = q_mass;
    nh.p_xi = 0.0;

    if comm.rank() == 0 {
        if let Some(ref gname) = config.group {
            println!(
                "Nose-Hoover: T*={}, tau={}, ndof={}, Q={}, group='{}'",
                t_target, tau, ndof, q_mass, gname
            );
        } else {
            println!(
                "Nose-Hoover: T*={}, tau={}, ndof={}, Q={}",
                t_target, tau, ndof, q_mass
            );
        }
    }
}

/// Fused pre-initial: half-step NH thermostat + Velocity Verlet initial integration.
/// 1. Compute KE (global reduction for MPI)
/// 2. Update p_xi by half step
/// 3. Fused loop: rescale velocities, half-kick, drift
pub fn nh_pre_initial(
    mut atoms: ResMut<Atom>,
    mut nh: ResMut<NoseHooverState>,
    comm: Res<CommResource>,
    groups: Res<GroupRegistry>,
) {
    let dt = atoms.dt;
    let mask = groups.mask_for(&nh.group_name);

    let ke_local = compute_ke(&atoms, mask);
    let ke = comm.all_reduce_sum_f64(ke_local);
    nh.p_xi += (dt / 2.0) * (2.0 * ke - nh.ndof * nh.target_temp);

    let scale = (-dt / 2.0 * nh.p_xi / nh.q_mass).exp();
    let nlocal = atoms.nlocal as usize;

    let inv_mass_ptr = atoms.inv_mass.as_ptr();
    let force_ptr = atoms.force.as_ptr();
    let vel_ptr = atoms.vel.as_mut_ptr();
    let pos_ptr = atoms.pos.as_mut_ptr();

    for i in 0..nlocal {
        unsafe {
            let v = &mut *vel_ptr.add(i);
            // Rescale (thermostat group only)
            if group_includes(mask, i) {
                v[0] *= scale;
                v[1] *= scale;
                v[2] *= scale;
            }
            // Half-kick
            let half_dt_over_m = 0.5 * dt * *inv_mass_ptr.add(i);
            let f = &*force_ptr.add(i);
            v[0] += half_dt_over_m * f[0];
            v[1] += half_dt_over_m * f[1];
            v[2] += half_dt_over_m * f[2];
            // Drift
            let p = &mut *pos_ptr.add(i);
            p[0] += v[0] * dt;
            p[1] += v[1] * dt;
            p[2] += v[2] * dt;
        }
    }
}

/// Fused post-final: Velocity Verlet final integration + half-step NH thermostat.
/// 1. Fused loop: half-kick, then rescale velocities
/// 2. Recompute KE (global reduction for MPI)
/// 3. Update p_xi by half step
pub fn nh_post_final(
    mut atoms: ResMut<Atom>,
    mut nh: ResMut<NoseHooverState>,
    comm: Res<CommResource>,
    groups: Res<GroupRegistry>,
) {
    let dt = atoms.dt;
    let mask = groups.mask_for(&nh.group_name);

    let scale = (-dt / 2.0 * nh.p_xi / nh.q_mass).exp();
    let nlocal = atoms.nlocal as usize;

    let inv_mass_ptr = atoms.inv_mass.as_ptr();
    let force_ptr = atoms.force.as_ptr();
    let vel_ptr = atoms.vel.as_mut_ptr();

    for i in 0..nlocal {
        unsafe {
            let v = &mut *vel_ptr.add(i);
            // Half-kick
            let half_dt_over_m = 0.5 * dt * *inv_mass_ptr.add(i);
            let f = &*force_ptr.add(i);
            v[0] += half_dt_over_m * f[0];
            v[1] += half_dt_over_m * f[1];
            v[2] += half_dt_over_m * f[2];
            // Rescale (thermostat group only)
            if group_includes(mask, i) {
                v[0] *= scale;
                v[1] *= scale;
                v[2] *= scale;
            }
        }
    }

    let ke_local = compute_ke(&atoms, mask);
    let ke = comm.all_reduce_sum_f64(ke_local);
    nh.p_xi += (dt / 2.0) * (2.0 * ke - nh.ndof * nh.target_temp);
}

// ═══════════════════════════════════════════════════════════════════════════
// Langevin thermostat
// ═══════════════════════════════════════════════════════════════════════════

fn default_damping() -> f64 {
    1.0
}
fn default_seed() -> u64 {
    12345
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[langevin]` — Langevin thermostat settings.
pub struct LangevinConfig {
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_damping")]
    pub damping: f64,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub group: Option<String>,
}

impl Default for LangevinConfig {
    fn default() -> Self {
        LangevinConfig {
            temperature: 0.85,
            damping: 1.0,
            seed: 12345,
            group: None,
        }
    }
}

pub struct LangevinState {
    pub rng: ChaCha8Rng,
    pub group_name: Option<String>,
    pub temperature: f64,
    pub damping: f64,
}

impl Default for LangevinState {
    fn default() -> Self {
        LangevinState {
            rng: ChaCha8Rng::seed_from_u64(12345),
            group_name: None,
            temperature: 0.85,
            damping: 1.0,
        }
    }
}

/// Langevin thermostat plugin: stochastic friction + random force.
pub struct LangevinPlugin;

impl Plugin for LangevinPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[langevin]
temperature = 0.85   # target T* (reduced units)
damping = 1.0        # friction coefficient gamma
seed = 12345         # RNG seed
# group = "mobile"  # optional: only thermostat this group"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<LangevinConfig>(app, "langevin");
        app.add_resource(LangevinState::default())
            .add_setup_system(setup_langevin, ScheduleSetupSet::PostSetup)
            .add_update_system(langevin_force, ScheduleSet::PostForce);
    }
}

pub fn setup_langevin(
    config: Res<LangevinConfig>,
    comm: Res<CommResource>,
    groups: Res<GroupRegistry>,
    mut state: ResMut<LangevinState>,
    scheduler_manager: Res<SchedulerManager>,
) {
    if scheduler_manager.index != 0 {
        return;
    }

    if let Some(ref gname) = config.group {
        groups.validate_name(gname, "Langevin thermostat");
    }

    // Seed RNG per-rank to get different random numbers on each process
    let rank_seed = config.seed.wrapping_add(comm.rank() as u64 * 1_000_000_007);
    state.rng = ChaCha8Rng::seed_from_u64(rank_seed);
    state.temperature = config.temperature;
    state.damping = config.damping;
    state.group_name = config.group.clone();

    if comm.rank() == 0 {
        if let Some(ref gname) = config.group {
            println!(
                "Langevin: T*={}, gamma={}, seed={}, group='{}'",
                config.temperature, config.damping, config.seed, gname
            );
        } else {
            println!(
                "Langevin: T*={}, gamma={}, seed={}",
                config.temperature, config.damping, config.seed
            );
        }
    }
}

/// Apply Langevin drag + random force at PostForce.
///
/// f_drag = -gamma * m * v
/// f_rand = sqrt(2 * gamma * m * kT / dt) * N(0,1)
pub fn langevin_force(
    mut atoms: ResMut<Atom>,
    mut state: ResMut<LangevinState>,
    groups: Res<GroupRegistry>,
) {
    let dt = atoms.dt;
    let gamma = state.damping;
    let kt = state.temperature;
    let nlocal = atoms.nlocal as usize;

    if gamma == 0.0 {
        return;
    }

    let mask = groups.mask_for(&state.group_name);

    for i in 0..nlocal {
        if !group_includes(mask, i) {
            continue;
        }
        let m = atoms.mass[i];
        let drag_coeff = gamma * m;
        let rand_coeff = (2.0 * drag_coeff * kt / dt).sqrt();

        // Drag force
        atoms.force[i][0] -= drag_coeff * atoms.vel[i][0];
        atoms.force[i][1] -= drag_coeff * atoms.vel[i][1];
        atoms.force[i][2] -= drag_coeff * atoms.vel[i][2];

        // Random force (3 components, normal distribution)
        atoms.force[i][0] += rand_coeff * normal_sample(&mut state.rng);
        atoms.force[i][1] += rand_coeff * normal_sample(&mut state.rng);
        atoms.force[i][2] += rand_coeff * normal_sample(&mut state.rng);
    }
}

/// Sample from standard normal distribution using Box-Muller transform.
fn normal_sample(rng: &mut ChaCha8Rng) -> f64 {
    use std::f64::consts::TAU;
    let u1: f64 = rng.gen::<f64>();
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.max(1e-300).ln()).sqrt() * (TAU * u2).cos()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_test_utils::make_single_comm;

    fn push_atom(atom: &mut Atom, tag: u32) {
        atom.push_test_atom(tag, [0.0; 3], 0.5, 1.0);
    }

    fn make_nh_app(n: usize, velocities: &[(f64, f64, f64)], t_target: f64, tau: f64) -> App {
        let mut app = App::new();

        let ndof = 3.0 * n as f64 - 3.0;
        let q_mass = ndof * t_target * tau * tau;

        let config = ThermostatConfig {
            temperature: t_target,
            coupling: tau,
            group: None,
        };
        let nh = NoseHooverState {
            p_xi: 0.0,
            q_mass,
            target_temp: t_target,
            ndof,
            group_name: None,
        };

        let mut atom = Atom::new();
        atom.dt = 0.001;
        for i in 0..n {
            push_atom(&mut atom, i as u32);
            atom.vel[i][0] = velocities[i].0;
            atom.vel[i][1] = velocities[i].1;
            atom.vel[i][2] = velocities[i].2;
        }
        atom.nlocal = n as u32;
        atom.natoms = n as u64;

        app.add_resource(config);
        app.add_resource(nh);
        app.add_resource(atom);
        app.add_resource(make_single_comm());
        app.add_resource(GroupRegistry::default());
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
            compute_ke(&a, None)
        };

        for _ in 0..1000 {
            app.run();
        }

        let ke_after = {
            let a = app.get_resource_ref::<Atom>().unwrap();
            compute_ke(&a, None)
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

    fn make_langevin_app(
        n: usize,
        velocities: &[(f64, f64, f64)],
        t_target: f64,
        gamma: f64,
        seed: u64,
    ) -> App {
        let mut app = App::new();

        let config = LangevinConfig {
            temperature: t_target,
            damping: gamma,
            seed,
            group: None,
        };

        let mut state = LangevinState::default();
        state.rng = ChaCha8Rng::seed_from_u64(seed);
        state.temperature = t_target;
        state.damping = gamma;

        let mut atom = Atom::new();
        atom.dt = 0.005;
        for i in 0..n {
            push_atom(&mut atom, i as u32);
            atom.vel[i][0] = velocities[i].0;
            atom.vel[i][1] = velocities[i].1;
            atom.vel[i][2] = velocities[i].2;
        }
        atom.nlocal = n as u32;
        atom.natoms = n as u64;

        app.add_resource(config);
        app.add_resource(state);
        app.add_resource(atom);
        app.add_resource(make_single_comm());
        app.add_resource(GroupRegistry::default());
        app.add_update_system(langevin_force, ScheduleSet::PostForce);
        app.organize_systems();
        app
    }

    #[test]
    fn langevin_applies_forces() {
        let n = 100;
        let vels: Vec<_> = (0..n).map(|_| (10.0, 0.0, 0.0)).collect();
        let mut app = make_langevin_app(n, &vels, 1.0, 10.0, 42);

        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        // With high gamma and velocity, drag dominates random force.
        // Drag = -gamma*m*v = -10*1*10 = -100 per atom, total = -10000
        // Random ~ sqrt(2*gamma*m*kT/dt) * N(0,1), averages to 0 over 100 atoms
        let mut total_fx = 0.0;
        for i in 0..n {
            total_fx += a.force[i][0];
        }
        assert!(total_fx < 0.0, "Drag should dominate: total_fx={}", total_fx);
    }

    #[test]
    fn langevin_zero_damping_no_effect() {
        let n = 5;
        let vels: Vec<_> = (0..n).map(|_| (2.0, 3.0, -1.0)).collect();
        let mut app = make_langevin_app(n, &vels, 1.0, 0.0, 99);

        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        // gamma=0 → no forces applied
        for i in 0..n {
            assert!(
                a.force[i][0].abs() < 1e-12,
                "Expected zero force at atom {}: {}",
                i,
                a.force[i][0]
            );
        }
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
        let ke = compute_ke(&a, None);
        assert!(ke.is_finite(), "KE should be finite: {}", ke);
        assert!(
            nh_state.p_xi.is_finite(),
            "p_xi should be finite: {}",
            nh_state.p_xi
        );
    }
}

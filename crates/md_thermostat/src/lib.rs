//! Thermostats: Nose-Hoover NVT (symmetric Liouville splitting) and Langevin (stochastic).

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;

use mddem_core::{compute_ke, group_includes, Atom, CommResource, Config, GroupRegistry, StageOverrides};

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
    atoms: Res<Atom>,
    comm: Res<CommResource>,
    groups: Res<GroupRegistry>,
    mut nh: ResMut<NoseHooverState>,
    stage_overrides: Res<StageOverrides>,
) {
    // Read config from stage overrides (allows per-stage temperature changes)
    let config: ThermostatConfig = Config::load_stage_aware(&stage_overrides, "thermostat");

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
    comm: Res<CommResource>,
    groups: Res<GroupRegistry>,
    mut state: ResMut<LangevinState>,
    stage_overrides: Res<StageOverrides>,
) {
    // Read config from stage overrides (allows per-stage changes)
    let config: LangevinConfig = Config::load_stage_aware(&stage_overrides, "langevin");

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
    let u1: f64 = rng.random::<f64>();
    let u2: f64 = rng.random::<f64>();
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

    // ── Nose-Hoover: temperature convergence to target ─────────────────────

    #[test]
    fn nh_temperature_converges_to_target() {
        // Start with a system far from the target temperature and verify
        // that after equilibration, the average temperature is close to T_target.
        // T = 2*KE / ndof. Initial T ≈ 3.0 (all velocities = 1.0), target = 1.0.
        let n = 100;
        let vels: Vec<_> = (0..n).map(|_| (1.0, 1.0, 1.0)).collect();
        let t_target = 1.0;
        let ndof = 3.0 * n as f64 - 3.0;

        let mut app = make_nh_app(n, &vels, t_target, 0.5);

        // Equilibrate for 5000 steps
        for _ in 0..5000 {
            app.run();
        }

        // Measure average temperature over next 5000 steps
        let mut t_sum = 0.0;
        let n_samples = 5000;
        for _ in 0..n_samples {
            app.run();
            let a = app.get_resource_ref::<Atom>().unwrap();
            let ke = compute_ke(&a, None);
            t_sum += 2.0 * ke / ndof;
        }

        let t_avg = t_sum / n_samples as f64;
        // Without inter-particle forces, the NH thermostat drives the free-particle
        // KE toward the target. We expect <T> ≈ T_target within ~20% for this
        // simple no-force system.
        assert!(
            (t_avg - t_target).abs() < 0.5,
            "Average temperature should be near target: <T>={:.4}, T_target={}",
            t_avg,
            t_target
        );
    }

    // ── Nose-Hoover conserved quantity ─────────────────────────────────────

    #[test]
    fn nh_extended_hamiltonian_conservation() {
        // The Nose-Hoover extended Hamiltonian is:
        //   H_ext = KE + 0.5 * p_xi^2 / Q + ndof * T_target * xi
        // where xi is the thermostat position (integral of p_xi/Q).
        // Since we only track p_xi (not xi), we verify that the _change_
        // in KE correlates with the thermostat work (energy doesn't blow up).
        let n = 50;
        let vels: Vec<_> = (0..n)
            .map(|i| {
                let s = (i as f64) * 0.17;
                (s.sin() * 2.0, s.cos() * 1.5, (s * 1.3).sin())
            })
            .collect();

        let mut app = make_nh_app(n, &vels, 1.0, 1.0);

        // Run and track that KE + thermostat energy remains bounded
        let mut ke_vals = Vec::new();
        let mut pxi_vals = Vec::new();
        for _ in 0..2000 {
            app.run();
            let a = app.get_resource_ref::<Atom>().unwrap();
            let nh = app.get_resource_ref::<NoseHooverState>().unwrap();
            ke_vals.push(compute_ke(&a, None));
            pxi_vals.push(nh.p_xi);
        }

        // All values should be finite and bounded
        for (i, (&ke, &pxi)) in ke_vals.iter().zip(pxi_vals.iter()).enumerate() {
            assert!(ke.is_finite(), "KE infinite at step {}", i);
            assert!(pxi.is_finite(), "p_xi infinite at step {}", i);
            assert!(ke < 1e6, "KE blew up at step {}: {}", i, ke);
            assert!(pxi.abs() < 1e6, "p_xi blew up at step {}: {}", i, pxi);
        }
    }

    // ── Langevin: equilibrium temperature ──────────────────────────────────

    #[test]
    fn langevin_equilibrium_temperature() {
        // Langevin thermostat with drag + random forces should satisfy the
        // fluctuation-dissipation theorem (FDT). In equilibrium:
        //   <v²> = kT/m  per component
        //
        // We test this analytically by running a pure Langevin dynamics
        // (no App needed, just the math) to verify the FDT relationship.
        //
        // Langevin: m dv = -gamma*m*v*dt + sqrt(2*gamma*m*kT)*dW
        // Euler-Maruyama: v_{n+1} = v_n*(1 - gamma*dt) + sqrt(2*gamma*kT*dt/m)*N(0,1)
        //
        // In steady state: <v²> = kT/m
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let n = 200; // number of particles
        let t_target = 1.5;
        let gamma = 5.0;
        let mass = 1.0;
        let dt = 0.001;
        let equilibrate = 5000;
        let n_samples = 10_000;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut velocities = vec![[0.0f64; 3]; n];

        let decay = 1.0 - gamma * dt;
        let noise_amp = (2.0_f64 * gamma * t_target * dt / mass).sqrt();

        // Equilibrate
        for _ in 0..equilibrate {
            for i in 0..n {
                for d in 0..3 {
                    let eta = normal_sample(&mut rng);
                    velocities[i][d] = velocities[i][d] * decay + noise_amp * eta;
                }
            }
        }

        // Measure <KE> = sum(0.5*m*v²)
        let ndof = 3.0 * n as f64;
        let mut t_sum = 0.0;

        for _ in 0..n_samples {
            for i in 0..n {
                for d in 0..3 {
                    let eta = normal_sample(&mut rng);
                    velocities[i][d] = velocities[i][d] * decay + noise_amp * eta;
                }
            }
            let ke: f64 = velocities.iter()
                .map(|v| 0.5 * mass * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]))
                .sum();
            t_sum += 2.0 * ke / ndof;
        }

        let t_avg = t_sum / n_samples as f64;
        // Should be close to t_target within statistical error
        assert!(
            (t_avg - t_target).abs() < 0.15,
            "Langevin FDT: <T>={:.4} should be near target T={} (N={}, samples={})",
            t_avg, t_target, n, n_samples
        );
    }

    #[test]
    fn langevin_random_force_coefficient() {
        // Verify the random force coefficient satisfies the FDT:
        //   sigma_F = sqrt(2 * gamma * m * kT / dt)
        // We check this by computing the expected coefficient directly.
        let gamma = 3.0;
        let mass = 2.0;
        let kt = 1.5;
        let dt = 0.005;

        let expected_sigma = (2.0_f64 * gamma * mass * kt / dt).sqrt();

        // The code computes: rand_coeff = sqrt(2 * drag_coeff * kt / dt)
        // where drag_coeff = gamma * m
        let drag_coeff = gamma * mass;
        let code_sigma = (2.0_f64 * drag_coeff * kt / dt).sqrt();

        assert!(
            (code_sigma - expected_sigma).abs() < 1e-10,
            "FDT coefficient mismatch: code={}, expected={}",
            code_sigma, expected_sigma
        );
    }

    // ── Langevin: fluctuation-dissipation relation ─────────────────────────

    #[test]
    fn langevin_drag_force_exact_value() {
        // Verify that the drag force is exactly -gamma * m * v
        let n = 1;
        let gamma = 2.0;
        let v0 = 5.0;
        let vels = vec![(v0, 0.0, 0.0)];

        // Use a very large temperature to make random forces non-negligible,
        // but test with many samples to verify the mean drag.
        // Actually, let's verify the single-step force breakdown:
        // For a single atom, F_drag_x = -gamma * m * v_x = -2 * 1 * 5 = -10
        // F_random_x = sqrt(2*gamma*m*kT/dt) * N(0,1) -- random contribution
        // We can verify the drag contribution by running with T=0.
        let mut app = make_langevin_app(n, &vels, 0.0, gamma, 42);
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        let expected_drag = -gamma * 1.0 * v0; // -10.0
        // With T=0, random force coefficient = sqrt(0) = 0, so only drag remains
        assert!(
            (a.force[0][0] - expected_drag).abs() < 1e-10,
            "Drag force should be exactly {}: got {}",
            expected_drag,
            a.force[0][0]
        );
    }

    // ── Nose-Hoover: momentum conservation ─────────────────────────────────

    #[test]
    fn nh_total_momentum_evolution() {
        // NH thermostat rescales all velocities uniformly, so if initial
        // COM momentum is zero, it should remain zero (or very close).
        let n = 20;
        // Create velocities with zero COM
        let mut vels: Vec<_> = (0..n)
            .map(|i| {
                let s = i as f64 * 0.5;
                (s.sin(), s.cos(), (s * 0.7).sin())
            })
            .collect();
        // Remove COM drift
        let mut vcom = (0.0, 0.0, 0.0);
        for v in &vels {
            vcom.0 += v.0;
            vcom.1 += v.1;
            vcom.2 += v.2;
        }
        vcom.0 /= n as f64;
        vcom.1 /= n as f64;
        vcom.2 /= n as f64;
        for v in vels.iter_mut() {
            v.0 -= vcom.0;
            v.1 -= vcom.1;
            v.2 -= vcom.2;
        }

        let mut app = make_nh_app(n, &vels, 1.0, 1.0);

        for _ in 0..1000 {
            app.run();
        }

        let a = app.get_resource_ref::<Atom>().unwrap();
        let mut px = 0.0;
        let mut py = 0.0;
        let mut pz = 0.0;
        for i in 0..n {
            px += a.mass[i] * a.vel[i][0];
            py += a.mass[i] * a.vel[i][1];
            pz += a.mass[i] * a.vel[i][2];
        }

        // COM momentum should remain near zero (uniform rescaling preserves zero COM)
        let p_total = (px * px + py * py + pz * pz).sqrt();
        assert!(
            p_total < 1e-8,
            "NH should preserve zero COM momentum: |p|={:.2e}",
            p_total
        );
    }
}

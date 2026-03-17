//! Atom manipulation fixes for MDDEM simulations.
//!
//! This crate provides a collection of per-atom fixes that modify forces and
//! velocities during a simulation timestep. Each fix targets atoms in a named
//! group and is configured via TOML arrays in the simulation config file.
//!
//! # Available Fixes
//!
//! | Fix | TOML key | Description |
//! |-----|----------|-------------|
//! | [`AddForceDef`] | `[[addforce]]` | Adds a constant force vector to atoms |
//! | [`SetForceDef`] | `[[setforce]]` | Overwrites the force vector on atoms |
//! | [`MoveLinearDef`] | `[[move_linear]]` | Moves atoms at a constant velocity |
//! | [`FreezeDef`] | `[[freeze]]` | Zeros velocity and force (immobilizes atoms) |
//! | [`ViscousDef`] | `[[viscous]]` | Applies velocity-proportional damping (F = −γv) |
//! | [`GravityConfig`] | `[gravity]` | Applies gravitational body force (F = mg) |
//!
//! # Plugins
//!
//! - [`FixesPlugin`] — registers all group-based fixes (addforce, setforce, move_linear, freeze, viscous)
//! - [`GravityPlugin`] — registers the gravity body force
//!
//! # Schedule Ordering
//!
//! - `move_linear` runs in **PreInitialIntegration** (to set velocity before position update)
//!   and **PostForce** (to zero force so FinalIntegration doesn't alter velocity).
//! - `addforce`, `setforce`, `freeze`, and `viscous` all run in **PostForce**.
//! - `gravity` runs in **Force**.

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, CommResource, Config, GroupRegistry};
use mddem_print::Thermo;

// ── Config structs ─────────────────────────────────────────────────────────

fn default_zero() -> f64 {
    0.0
}

/// Adds a constant external force to every atom in the specified group.
///
/// The force components `(fx, fy, fz)` are **added** to each atom's existing
/// force every timestep during the PostForce schedule phase.
///
/// # TOML Configuration
///
/// ```toml
/// [[addforce]]
/// group = "fluid"   # (required) name of the atom group
/// fx = 0.1          # force in x direction (default: 0.0)
/// fy = 0.0          # force in y direction (default: 0.0)
/// fz = 0.0          # force in z direction (default: 0.0)
/// ```
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct AddForceDef {
    /// Name of the atom group this fix applies to.
    pub group: String,
    /// Force component in the x direction. Default: 0.0.
    #[serde(default = "default_zero")]
    pub fx: f64,
    /// Force component in the y direction. Default: 0.0.
    #[serde(default = "default_zero")]
    pub fy: f64,
    /// Force component in the z direction. Default: 0.0.
    #[serde(default = "default_zero")]
    pub fz: f64,
}

/// Overwrites the force on every atom in the specified group with a constant value.
///
/// Unlike [`AddForceDef`], this **replaces** the force rather than adding to it.
/// Useful for boundary atoms where computed forces should be ignored.
///
/// # TOML Configuration
///
/// ```toml
/// [[setforce]]
/// group = "wall"    # (required) name of the atom group
/// fx = 0.0          # force in x direction (default: 0.0)
/// fy = 0.0          # force in y direction (default: 0.0)
/// fz = 0.0          # force in z direction (default: 0.0)
/// ```
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct SetForceDef {
    /// Name of the atom group this fix applies to.
    pub group: String,
    /// Force component in the x direction. Default: 0.0.
    #[serde(default = "default_zero")]
    pub fx: f64,
    /// Force component in the y direction. Default: 0.0.
    #[serde(default = "default_zero")]
    pub fy: f64,
    /// Force component in the z direction. Default: 0.0.
    #[serde(default = "default_zero")]
    pub fz: f64,
}

/// Moves atoms in a group at a prescribed constant velocity.
///
/// Before the Verlet position update (PreInitialIntegration), this fix sets
/// the velocity of group atoms to `(vx, vy, vz)`. After forces are computed
/// (PostForce), it zeros the force so that FinalIntegration does not alter
/// the velocity. The result is that group atoms translate at a constant rate
/// regardless of applied forces.
///
/// # TOML Configuration
///
/// ```toml
/// [[move_linear]]
/// group = "piston"  # (required) name of the atom group
/// vx = 0.0          # velocity in x direction (default: 0.0)
/// vy = 0.0          # velocity in y direction (default: 0.0)
/// vz = -0.001       # velocity in z direction (default: 0.0)
/// ```
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct MoveLinearDef {
    /// Name of the atom group this fix applies to.
    pub group: String,
    /// Prescribed velocity in the x direction. Default: 0.0.
    #[serde(default = "default_zero")]
    pub vx: f64,
    /// Prescribed velocity in the y direction. Default: 0.0.
    #[serde(default = "default_zero")]
    pub vy: f64,
    /// Prescribed velocity in the z direction. Default: 0.0.
    #[serde(default = "default_zero")]
    pub vz: f64,
}

/// Freezes atoms in a group by zeroing both velocity and force every timestep.
///
/// This effectively immobilizes atoms — they remain at their initial positions
/// and do not respond to any forces. Useful for fixed boundary particles.
///
/// # TOML Configuration
///
/// ```toml
/// [[freeze]]
/// group = "frozen"   # (required) name of the atom group
/// ```
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct FreezeDef {
    /// Name of the atom group this fix applies to.
    pub group: String,
}

/// Applies velocity-proportional viscous damping to atoms in a group.
///
/// Adds a damping force **F** = −γ **v** to each atom, where γ (`gamma`) is
/// the damping coefficient and **v** is the atom's velocity. This dissipates
/// kinetic energy and is commonly used to reach static equilibrium in DEM
/// simulations.
///
/// # TOML Configuration
///
/// ```toml
/// [[viscous]]
/// group = "all"      # (required) name of the atom group
/// gamma = 0.1        # (required) damping coefficient (force/velocity units)
/// ```
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ViscousDef {
    /// Name of the atom group this fix applies to.
    pub group: String,
    /// Damping coefficient γ. Units: force / velocity.
    pub gamma: f64,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct NveLimitDef {
    pub group: String,
    pub max_displacement: f64,
}

// ── Registry ───────────────────────────────────────────────────────────────

/// Central storage for all configured fix definitions.
///
/// Populated at plugin build time from the TOML config and stored as an
/// [`App`] resource. Each system reads its corresponding vector from here.
pub struct FixesRegistry {
    /// All `[[addforce]]` definitions.
    pub add_forces: Vec<AddForceDef>,
    /// All `[[setforce]]` definitions.
    pub set_forces: Vec<SetForceDef>,
    /// All `[[move_linear]]` definitions.
    pub move_linears: Vec<MoveLinearDef>,
    /// All `[[freeze]]` definitions.
    pub freezes: Vec<FreezeDef>,
    /// All `[[viscous]]` definitions.
    pub viscous: Vec<ViscousDef>,
    pub nve_limit: Vec<NveLimitDef>,
}

// ── Plugin ─────────────────────────────────────────────────────────────────

/// Plugin that registers group-based atom fixes: addforce, setforce,
/// move_linear, freeze, and viscous damping.
///
/// Only registers update systems for fix types that have at least one
/// definition in the config, avoiding unnecessary per-timestep overhead.
pub struct FixesPlugin;

impl Plugin for FixesPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# [[addforce]]
# group = "fluid"
# fx = 0.1
# fy = 0.0
# fz = 0.0

# [[setforce]]
# group = "wall"
# fx = 0.0
# fy = 0.0
# fz = 0.0

# [[move_linear]]
# group = "piston"
# vx = 0.0
# vy = 0.0
# vz = -0.001

# [[freeze]]
# group = "frozen"

# [[nve_limit]]
# group = "all"
# max_displacement = 0.0001  # max distance any atom can move per step"#,
        )
    }

    fn build(&self, app: &mut App) {
        let config = app
            .get_resource_ref::<Config>()
            .expect("Config resource must exist before FixesPlugin");

        let registry = FixesRegistry {
            add_forces: config.parse_array::<AddForceDef>("addforce"),
            set_forces: config.parse_array::<SetForceDef>("setforce"),
            move_linears: config.parse_array::<MoveLinearDef>("move_linear"),
            freezes: config.parse_array::<FreezeDef>("freeze"),
            viscous: config.parse_array::<ViscousDef>("viscous"),
            nve_limit: config.parse_array::<NveLimitDef>("nve_limit"),
        };

        drop(config);

        let has_any = !registry.add_forces.is_empty()
            || !registry.set_forces.is_empty()
            || !registry.move_linears.is_empty()
            || !registry.freezes.is_empty()
            || !registry.viscous.is_empty()
            || !registry.nve_limit.is_empty();

        if !has_any {
            app.add_resource(registry);
            return;
        }

        let has_move = !registry.move_linears.is_empty();
        let has_add = !registry.add_forces.is_empty();
        let has_set = !registry.set_forces.is_empty();
        let has_freeze = !registry.freezes.is_empty();
        let has_viscous = !registry.viscous.is_empty();
        let has_nve_limit = !registry.nve_limit.is_empty();

        app.add_resource(registry)
            .add_setup_system(setup_fixes, ScheduleSetupSet::PostSetup);

        if has_move {
            app.add_update_system(apply_move_linear_pre, ScheduleSet::PreInitialIntegration);
            app.add_update_system(apply_move_linear_post, ScheduleSet::PostForce);
        }
        if has_add {
            app.add_update_system(apply_add_force, ScheduleSet::PostForce);
        }
        if has_set {
            app.add_update_system(apply_set_force, ScheduleSet::PostForce);
        }
        if has_freeze {
            app.add_update_system(apply_freeze, ScheduleSet::PostForce);
        }
        if has_viscous {
            app.add_update_system(apply_viscous, ScheduleSet::PostForce);
        }
        if has_nve_limit {
            app.add_update_system(apply_nve_limit, ScheduleSet::PostFinalIntegration);
        }
    }
}

// ── Systems ────────────────────────────────────────────────────────────────

/// Validates all fix group names at setup time and prints a summary on rank 0.
///
/// # Panics
///
/// Panics (via `GroupRegistry::validate_name`) if any fix references a group
/// name that has not been defined.
fn setup_fixes(registry: Res<FixesRegistry>, comm: Res<CommResource>, groups: Res<GroupRegistry>) {
    // Validate all group names at setup time.
    for f in &registry.add_forces {
        groups.validate_name(&f.group, "fix addforce");
    }
    for f in &registry.set_forces {
        groups.validate_name(&f.group, "fix setforce");
    }
    for f in &registry.move_linears {
        groups.validate_name(&f.group, "fix move_linear");
    }
    for f in &registry.freezes {
        groups.validate_name(&f.group, "fix freeze");
    }
    for f in &registry.viscous {
        groups.validate_name(&f.group, "fix viscous");
    }
    for f in &registry.nve_limit {
        groups.validate_name(&f.group, "fix nve_limit");
    }

    if comm.rank() != 0 {
        return;
    }

    // Print a summary of all active fixes on rank 0.
    for f in &registry.add_forces {
        println!(
            "Fix addforce: group='{}', fx={}, fy={}, fz={}",
            f.group, f.fx, f.fy, f.fz
        );
    }
    for f in &registry.set_forces {
        println!(
            "Fix setforce: group='{}', fx={}, fy={}, fz={}",
            f.group, f.fx, f.fy, f.fz
        );
    }
    for f in &registry.move_linears {
        println!(
            "Fix move_linear: group='{}', vx={}, vy={}, vz={}",
            f.group, f.vx, f.vy, f.vz
        );
    }
    for f in &registry.freezes {
        println!("Fix freeze: group='{}'", f.group);
    }
    for f in &registry.viscous {
        println!("Fix viscous: group='{}', gamma={}", f.group, f.gamma);
    }
    for f in &registry.nve_limit {
        println!(
            "Fix nve_limit: group='{}', max_displacement={}",
            f.group, f.max_displacement
        );
    }
}

/// Sets velocity to the prescribed constant **before** the Verlet position update,
/// so that atoms move at the exact prescribed rate during InitialIntegration.
fn apply_move_linear_pre(
    mut atoms: ResMut<Atom>,
    registry: Res<FixesRegistry>,
    groups: Res<GroupRegistry>,
) {
    let nlocal = atoms.nlocal as usize;
    for def in &registry.move_linears {
        let group = groups.expect(&def.group);
        for i in 0..nlocal {
            if group.mask[i] {
                atoms.vel[i][0] = def.vx;
                atoms.vel[i][1] = def.vy;
                atoms.vel[i][2] = def.vz;
            }
        }
    }
}

/// Adds a constant force `(fx, fy, fz)` to every atom in the group.
/// Applied during PostForce, so it accumulates on top of pair/bond forces.
fn apply_add_force(
    mut atoms: ResMut<Atom>,
    registry: Res<FixesRegistry>,
    groups: Res<GroupRegistry>,
) {
    let nlocal = atoms.nlocal as usize;
    for def in &registry.add_forces {
        let group = groups.expect(&def.group);
        for i in 0..nlocal {
            if group.mask[i] {
                atoms.force[i][0] += def.fx;
                atoms.force[i][1] += def.fy;
                atoms.force[i][2] += def.fz;
            }
        }
    }
}

/// Overwrites the force on every atom in the group with `(fx, fy, fz)`,
/// discarding any previously computed forces for those atoms.
fn apply_set_force(
    mut atoms: ResMut<Atom>,
    registry: Res<FixesRegistry>,
    groups: Res<GroupRegistry>,
) {
    let nlocal = atoms.nlocal as usize;
    for def in &registry.set_forces {
        let group = groups.expect(&def.group);
        for i in 0..nlocal {
            if group.mask[i] {
                atoms.force[i][0] = def.fx;
                atoms.force[i][1] = def.fy;
                atoms.force[i][2] = def.fz;
            }
        }
    }
}

/// Zeros both velocity and force on every frozen atom, keeping them immobile.
fn apply_freeze(
    mut atoms: ResMut<Atom>,
    registry: Res<FixesRegistry>,
    groups: Res<GroupRegistry>,
) {
    let nlocal = atoms.nlocal as usize;
    for def in &registry.freezes {
        let group = groups.expect(&def.group);
        for i in 0..nlocal {
            if group.mask[i] {
                atoms.vel[i][0] = 0.0;
                atoms.vel[i][1] = 0.0;
                atoms.vel[i][2] = 0.0;
                atoms.force[i][0] = 0.0;
                atoms.force[i][1] = 0.0;
                atoms.force[i][2] = 0.0;
            }
        }
    }
}

/// Zeros force on move_linear atoms after force computation, preventing
/// FinalIntegration from altering their prescribed velocity.
fn apply_move_linear_post(
    mut atoms: ResMut<Atom>,
    registry: Res<FixesRegistry>,
    groups: Res<GroupRegistry>,
) {
    let nlocal = atoms.nlocal as usize;
    for def in &registry.move_linears {
        let group = groups.expect(&def.group);
        for i in 0..nlocal {
            if group.mask[i] {
                atoms.force[i][0] = 0.0;
                atoms.force[i][1] = 0.0;
                atoms.force[i][2] = 0.0;
            }
        }
    }
}

/// Applies velocity-proportional viscous damping: **F**_damp = −γ **v**.
/// Subtracts `gamma * velocity` from each force component of group atoms.
fn apply_viscous(
    mut atoms: ResMut<Atom>,
    registry: Res<FixesRegistry>,
    groups: Res<GroupRegistry>,
) {
    let nlocal = atoms.nlocal as usize;
    for def in &registry.viscous {
        let group = groups.expect(&def.group);
        let gamma = def.gamma;
        for i in 0..nlocal {
            if group.mask[i] {
                atoms.force[i][0] -= gamma * atoms.vel[i][0];
                atoms.force[i][1] -= gamma * atoms.vel[i][1];
                atoms.force[i][2] -= gamma * atoms.vel[i][2];
            }
        }
    }
}

/// Cap maximum displacement per timestep by scaling velocity.
/// Preserves direction; only reduces magnitude when `|v| * dt > max_displacement`.
fn apply_nve_limit(
    mut atoms: ResMut<Atom>,
    registry: Res<FixesRegistry>,
    groups: Res<GroupRegistry>,
    mut thermo: Option<ResMut<Thermo>>,
) {
    let nlocal = atoms.nlocal as usize;
    let dt = atoms.dt;
    let mut n_limited: usize = 0;
    for def in &registry.nve_limit {
        let group = groups.expect(&def.group);
        let vmax = def.max_displacement / dt;
        for i in 0..nlocal {
            if !group.mask[i] {
                continue;
            }
            let vx = atoms.vel[i][0];
            let vy = atoms.vel[i][1];
            let vz = atoms.vel[i][2];
            let vmag = (vx * vx + vy * vy + vz * vz).sqrt();
            if vmag > vmax {
                let scale = vmax / vmag;
                atoms.vel[i][0] *= scale;
                atoms.vel[i][1] *= scale;
                atoms.vel[i][2] *= scale;
                n_limited += 1;
            }
        }
    }
    if let Some(ref mut t) = thermo {
        t.set("n_limited", n_limited as f64);
    }
}

// ── Gravity ─────────────────────────────────────────────────────────────────

/// Gravitational acceleration configuration.
///
/// Applies a body force **F** = m **g** to every local atom each timestep,
/// where m is the atom's mass and **g** = `(gx, gy, gz)`.
///
/// # TOML Configuration
///
/// ```toml
/// [gravity]
/// gx = 0.0      # acceleration in x direction (default: 0.0)
/// gy = 0.0      # acceleration in y direction (default: 0.0)
/// gz = -9.81    # acceleration in z direction (default: -9.81)
/// ```
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct GravityConfig {
    /// Gravitational acceleration in x direction. Default: 0.0.
    #[serde(default)]
    pub gx: f64,
    /// Gravitational acceleration in y direction. Default: 0.0.
    #[serde(default)]
    pub gy: f64,
    /// Gravitational acceleration in z direction. Default: −9.81.
    #[serde(default = "default_gravity_gz")]
    pub gz: f64,
}

impl Default for GravityConfig {
    fn default() -> Self {
        GravityConfig {
            gx: 0.0,
            gy: 0.0,
            gz: -9.81,
        }
    }
}

fn default_gravity_gz() -> f64 {
    -9.81
}

/// Plugin that applies a constant gravitational body force (**F** = m**g**)
/// to all local atoms during the Force schedule phase.
pub struct GravityPlugin;

impl Plugin for GravityPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[gravity]
# Gravitational acceleration components (m/s^2)
gx = 0.0
gy = 0.0
gz = -9.81"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<GravityConfig>(app, "gravity");
        app.add_update_system(apply_gravity, ScheduleSet::Force);
    }
}

/// Applies gravitational body force **F**_i = m_i **g** to all local atoms.
///
/// Ghost atoms (index ≥ `nlocal`) are not affected.
pub fn apply_gravity(mut atoms: ResMut<Atom>, gravity: Res<GravityConfig>) {
    for i in 0..atoms.nlocal as usize {
        atoms.force[i][0] += atoms.mass[i] * gravity.gx;
        atoms.force[i][1] += atoms.mass[i] * gravity.gy;
        atoms.force[i][2] += atoms.mass[i] * gravity.gz;
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_test_utils::{make_atoms, make_group_registry};

    #[test]
    fn test_addforce_applies_constant_force() {
        let mut atoms = make_atoms(3);
        let groups = make_group_registry("fluid", vec![true, false, true]);
        let registry = FixesRegistry {
            add_forces: vec![AddForceDef {
                group: "fluid".to_string(),
                fx: 1.5,
                fy: 0.0,
                fz: -0.5,
            }],
            set_forces: vec![],
            move_linears: vec![],
            freezes: vec![],
            viscous: vec![],
            nve_limit: vec![],
        };

        // Set some initial force
        atoms.force[0][0] = 2.0;
        atoms.force[2][0] = 3.0;

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_update_system(apply_add_force, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        assert!((a.force[0][0] - 3.5).abs() < 1e-12); // 2.0 + 1.5
        assert!((a.force[1][0]).abs() < 1e-12); // not in group
        assert!((a.force[2][0] - 4.5).abs() < 1e-12); // 3.0 + 1.5
        assert!((a.force[0][2] - (-0.5)).abs() < 1e-12);
        assert!((a.force[1][2]).abs() < 1e-12);
    }

    #[test]
    fn test_setforce_overrides_force() {
        let mut atoms = make_atoms(2);
        atoms.force[0][0] = 100.0;
        atoms.force[0][1] = 200.0;
        atoms.force[0][2] = 300.0;

        let groups = make_group_registry("wall", vec![true, false]);
        let registry = FixesRegistry {
            add_forces: vec![],
            set_forces: vec![SetForceDef {
                group: "wall".to_string(),
                fx: 1.0,
                fy: 2.0,
                fz: 3.0,
            }],
            move_linears: vec![],
            freezes: vec![],
            viscous: vec![],
            nve_limit: vec![],
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_update_system(apply_set_force, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        assert!((a.force[0][0] - 1.0).abs() < 1e-12);
        assert!((a.force[0][1] - 2.0).abs() < 1e-12);
        assert!((a.force[0][2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_freeze_zeros_vel_and_force() {
        let mut atoms = make_atoms(3);
        atoms.vel[1][0] = 5.0;
        atoms.vel[1][1] = 6.0;
        atoms.vel[1][2] = 7.0;
        atoms.force[1][0] = 10.0;
        atoms.force[1][1] = 20.0;
        atoms.force[1][2] = 30.0;

        let groups = make_group_registry("frozen", vec![false, true, false]);
        let registry = FixesRegistry {
            add_forces: vec![],
            set_forces: vec![],
            move_linears: vec![],
            freezes: vec![FreezeDef {
                group: "frozen".to_string(),
            }],
            viscous: vec![],
            nve_limit: vec![],
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_update_system(apply_freeze, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        assert!((a.vel[1][0]).abs() < 1e-12);
        assert!((a.vel[1][1]).abs() < 1e-12);
        assert!((a.vel[1][2]).abs() < 1e-12);
        assert!((a.force[1][0]).abs() < 1e-12);
        assert!((a.force[1][1]).abs() < 1e-12);
        assert!((a.force[1][2]).abs() < 1e-12);
    }

    #[test]
    fn test_move_linear_constant_velocity() {
        let atoms = make_atoms(2);
        let groups = make_group_registry("piston", vec![true, false]);
        let registry = FixesRegistry {
            add_forces: vec![],
            set_forces: vec![],
            move_linears: vec![MoveLinearDef {
                group: "piston".to_string(),
                vx: 0.0,
                vy: 0.0,
                vz: -0.5,
            }],
            freezes: vec![],
            viscous: vec![],
            nve_limit: vec![],
        };

        // Pre step: sets velocity
        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_update_system(apply_move_linear_pre, ScheduleSet::PreInitialIntegration);
        app.add_update_system(apply_move_linear_post, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        assert!((a.vel[0][2] - (-0.5)).abs() < 1e-12);
        assert!((a.vel[1][2]).abs() < 1e-12); // not in group
        assert!((a.force[0][0]).abs() < 1e-12); // force zeroed by post
        assert!((a.force[0][2]).abs() < 1e-12);
    }

    // ── Viscous tests ──────────────────────────────────────────────────────

    #[test]
    fn test_viscous_opposes_velocity() {
        let mut atoms = make_atoms(2);
        atoms.vel[0][0] = 1.0;
        atoms.vel[0][1] = -2.0;
        atoms.vel[0][2] = 0.5;

        let groups = make_group_registry("all", vec![true, true]);
        let registry = FixesRegistry {
            add_forces: vec![],
            set_forces: vec![],
            move_linears: vec![],
            freezes: vec![],
            viscous: vec![ViscousDef {
                group: "all".to_string(),
                gamma: 0.1,
            }],
            nve_limit: vec![],
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_update_system(apply_viscous, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        assert!((a.force[0][0] - (-0.1)).abs() < 1e-12, "fx = -gamma*vx");
        assert!((a.force[0][1] - 0.2).abs() < 1e-12, "fy = -gamma*vy");
        assert!((a.force[0][2] - (-0.05)).abs() < 1e-12, "fz = -gamma*vz");
    }

    #[test]
    fn test_viscous_zero_at_rest() {
        let atoms = make_atoms(2); // velocities are 0
        let groups = make_group_registry("all", vec![true, true]);
        let registry = FixesRegistry {
            add_forces: vec![],
            set_forces: vec![],
            move_linears: vec![],
            freezes: vec![],
            viscous: vec![ViscousDef {
                group: "all".to_string(),
                gamma: 0.1,
            }],
            nve_limit: vec![],
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_update_system(apply_viscous, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        assert!((a.force[0][0]).abs() < 1e-15);
        assert!((a.force[0][1]).abs() < 1e-15);
        assert!((a.force[0][2]).abs() < 1e-15);
    }

    // ── Gravity tests ──────────────────────────────────────────────────────

    fn make_gravity_atom(mass: f64) -> Atom {
        let mut atom = Atom::new();
        atom.dt = 1e-6;
        atom.push_test_atom(0, [0.0; 3], 0.001, mass);
        atom.nlocal = 1;
        atom.natoms = 1;
        atom
    }

    #[test]
    fn gravity_applies_force_equal_to_mg() {
        let mass = 0.5;
        let gz = -9.81;

        let mut app = App::new();
        app.add_resource(make_gravity_atom(mass));
        app.add_resource(GravityConfig {
            gx: 0.0,
            gy: 0.0,
            gz,
        });
        app.add_update_system(apply_gravity, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!((atom.force[0][0]).abs() < 1e-15);
        assert!((atom.force[0][1]).abs() < 1e-15);
        assert!((atom.force[0][2] - mass * gz).abs() < 1e-15);
    }

    #[test]
    fn gravity_skips_ghost_atoms() {
        let mass = 1.0;
        let gz = -9.81;

        let mut atom = make_gravity_atom(mass);
        // Add a ghost atom
        atom.push_test_atom(1, [0.0; 3], 0.001, mass);
        atom.is_ghost[1] = true;
        // nlocal stays 1, ghost is index 1

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(GravityConfig {
            gx: 0.0,
            gy: 0.0,
            gz,
        });
        app.add_update_system(apply_gravity, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Local atom gets force
        assert!((atom.force[0][2] - mass * gz).abs() < 1e-15);
        // Ghost atom does not
        assert!((atom.force[1][2]).abs() < 1e-15);
    }

    // ── NVE/Limit tests ─────────────────────────────────────────────────

    fn make_nve_limit_registry(group: &str, max_displacement: f64) -> FixesRegistry {
        FixesRegistry {
            add_forces: vec![],
            set_forces: vec![],
            move_linears: vec![],
            freezes: vec![],
            viscous: vec![],
            nve_limit: vec![NveLimitDef {
                group: group.to_string(),
                max_displacement,
            }],
        }
    }

    #[test]
    fn nve_limit_caps_high_velocity() {
        let mut atoms = make_atoms(1);
        atoms.dt = 0.001;
        // Velocity of 100 → displacement = 100 * 0.001 = 0.1 per step
        atoms.vel[0] = [100.0, 0.0, 0.0];

        let max_d = 0.01; // limit to 0.01 per step
        let groups = make_group_registry("all", vec![true]);
        let registry = make_nve_limit_registry("all", max_d);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_update_system(apply_nve_limit, ScheduleSet::PostFinalIntegration);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        let vmag = (a.vel[0][0].powi(2) + a.vel[0][1].powi(2) + a.vel[0][2].powi(2)).sqrt();
        let displacement = vmag * a.dt;
        assert!(
            (displacement - max_d).abs() < 1e-12,
            "displacement {} should equal max_displacement {}",
            displacement,
            max_d
        );
    }

    #[test]
    fn nve_limit_does_not_change_small_velocity() {
        let mut atoms = make_atoms(1);
        atoms.dt = 0.001;
        // Velocity of 1.0 → displacement = 0.001 per step, well under limit
        atoms.vel[0] = [0.6, 0.8, 0.0];

        let max_d = 0.01;
        let groups = make_group_registry("all", vec![true]);
        let registry = make_nve_limit_registry("all", max_d);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_update_system(apply_nve_limit, ScheduleSet::PostFinalIntegration);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        assert!((a.vel[0][0] - 0.6).abs() < 1e-15);
        assert!((a.vel[0][1] - 0.8).abs() < 1e-15);
        assert!((a.vel[0][2]).abs() < 1e-15);
    }

    #[test]
    fn nve_limit_preserves_direction() {
        let mut atoms = make_atoms(1);
        atoms.dt = 0.001;
        atoms.vel[0] = [3.0, 4.0, 0.0]; // magnitude = 5.0, displacement = 0.005

        let max_d = 0.001; // limit to 0.001 → vmax = 1.0
        let groups = make_group_registry("all", vec![true]);
        let registry = make_nve_limit_registry("all", max_d);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_update_system(apply_nve_limit, ScheduleSet::PostFinalIntegration);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        // Direction should be (3/5, 4/5, 0) = (0.6, 0.8, 0)
        let vmag = (a.vel[0][0].powi(2) + a.vel[0][1].powi(2) + a.vel[0][2].powi(2)).sqrt();
        assert!((vmag - 1.0).abs() < 1e-12, "vmag should be 1.0, got {}", vmag);
        assert!((a.vel[0][0] / vmag - 0.6).abs() < 1e-12, "direction x preserved");
        assert!((a.vel[0][1] / vmag - 0.8).abs() < 1e-12, "direction y preserved");
    }

    #[test]
    fn nve_limit_zero_velocity_no_panic() {
        let mut atoms = make_atoms(1);
        atoms.dt = 0.001;
        atoms.vel[0] = [0.0, 0.0, 0.0];

        let groups = make_group_registry("all", vec![true]);
        let registry = make_nve_limit_registry("all", 0.01);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_update_system(apply_nve_limit, ScheduleSet::PostFinalIntegration);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        assert!((a.vel[0][0]).abs() < 1e-15);
        assert!((a.vel[0][1]).abs() < 1e-15);
        assert!((a.vel[0][2]).abs() < 1e-15);
    }

    #[test]
    fn nve_limit_respects_group_filter() {
        let mut atoms = make_atoms(2);
        atoms.dt = 0.001;
        atoms.vel[0] = [100.0, 0.0, 0.0]; // in group, should be capped
        atoms.vel[1] = [100.0, 0.0, 0.0]; // not in group, unchanged

        let groups = make_group_registry("limited", vec![true, false]);
        let registry = make_nve_limit_registry("limited", 0.01);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_update_system(apply_nve_limit, ScheduleSet::PostFinalIntegration);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        // Atom 0: capped to 0.01 / 0.001 = 10.0
        assert!((a.vel[0][0] - 10.0).abs() < 1e-12, "atom 0 should be capped");
        // Atom 1: unchanged at 100.0
        assert!((a.vel[1][0] - 100.0).abs() < 1e-12, "atom 1 should be unchanged");
    }
}

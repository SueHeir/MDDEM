//! Atom manipulation fixes: addforce, setforce, freeze, move_linear.

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, AtomDataRegistry, CommResource, Config, GroupRegistry};

// ── Config structs ─────────────────────────────────────────────────────────

fn default_zero() -> f64 {
    0.0
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct AddForceDef {
    pub group: String,
    #[serde(default = "default_zero")]
    pub fx: f64,
    #[serde(default = "default_zero")]
    pub fy: f64,
    #[serde(default = "default_zero")]
    pub fz: f64,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct SetForceDef {
    pub group: String,
    #[serde(default = "default_zero")]
    pub fx: f64,
    #[serde(default = "default_zero")]
    pub fy: f64,
    #[serde(default = "default_zero")]
    pub fz: f64,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct MoveLinearDef {
    pub group: String,
    #[serde(default = "default_zero")]
    pub vx: f64,
    #[serde(default = "default_zero")]
    pub vy: f64,
    #[serde(default = "default_zero")]
    pub vz: f64,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct FreezeDef {
    pub group: String,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ViscousDef {
    pub group: String,
    pub gamma: f64,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct RotationalViscousDef {
    pub group: String,
    pub gamma: f64,
}

// ── Registry ───────────────────────────────────────────────────────────────

pub struct FixesRegistry {
    pub add_forces: Vec<AddForceDef>,
    pub set_forces: Vec<SetForceDef>,
    pub move_linears: Vec<MoveLinearDef>,
    pub freezes: Vec<FreezeDef>,
    pub viscous: Vec<ViscousDef>,
    pub rotational_viscous: Vec<RotationalViscousDef>,
}

// ── Plugin ─────────────────────────────────────────────────────────────────

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

# [[viscous]]
# group = "all"
# gamma = 0.1  # translational damping coefficient (force = -gamma * velocity)

# [[rotational_viscous]]
# group = "all"
# gamma = 0.001  # angular damping coefficient (torque = -gamma * omega)"#,
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
            rotational_viscous: config.parse_array::<RotationalViscousDef>("rotational_viscous"),
        };

        drop(config);

        let has_any = !registry.add_forces.is_empty()
            || !registry.set_forces.is_empty()
            || !registry.move_linears.is_empty()
            || !registry.freezes.is_empty()
            || !registry.viscous.is_empty()
            || !registry.rotational_viscous.is_empty();

        if !has_any {
            app.add_resource(registry);
            return;
        }

        let has_move = !registry.move_linears.is_empty();
        let has_add = !registry.add_forces.is_empty();
        let has_set = !registry.set_forces.is_empty();
        let has_freeze = !registry.freezes.is_empty();
        let has_viscous = !registry.viscous.is_empty();
        let has_rotational_viscous = !registry.rotational_viscous.is_empty();

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
        if has_rotational_viscous {
            app.add_update_system(apply_rotational_viscous, ScheduleSet::PostForce);
        }
    }
}

// ── Systems ────────────────────────────────────────────────────────────────

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
    for f in &registry.rotational_viscous {
        groups.validate_name(&f.group, "fix rotational_viscous");
    }

    if comm.rank() != 0 {
        return;
    }
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
    for f in &registry.rotational_viscous {
        println!(
            "Fix rotational_viscous: group='{}', gamma={}",
            f.group, f.gamma
        );
    }
}

/// Set velocity = constant BEFORE Verlet position update, so positions move at prescribed rate.
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

/// Add a constant force to all atoms in the group.
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

/// Overwrite force on all atoms in the group.
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

/// Zero velocity and force on frozen atoms.
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

/// Zero force on move_linear atoms so FinalIntegration doesn't change velocity.
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

/// Apply velocity-proportional damping: F = -gamma * v.
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

/// Apply angular-velocity-proportional damping: τ = -gamma * ω.
fn apply_rotational_viscous(
    atoms: Res<Atom>,
    atom_data: Res<AtomDataRegistry>,
    registry: Res<FixesRegistry>,
    groups: Res<GroupRegistry>,
) {
    let mut dem = atom_data.expect_mut::<dem_atom::DemAtom>("apply_rotational_viscous");
    let nlocal = atoms.nlocal as usize;
    for def in &registry.rotational_viscous {
        let group = groups.expect(&def.group);
        let gamma = def.gamma;
        for i in 0..nlocal {
            if group.mask[i] {
                dem.torque[i][0] -= gamma * dem.omega[i][0];
                dem.torque[i][1] -= gamma * dem.omega[i][1];
                dem.torque[i][2] -= gamma * dem.omega[i][2];
            }
        }
    }
}

// ── Gravity ─────────────────────────────────────────────────────────────────

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[gravity]` — gravitational acceleration components.
pub struct GravityConfig {
    /// Gravity in x direction (m/s²).
    #[serde(default)]
    pub gx: f64,
    /// Gravity in y direction (m/s²).
    #[serde(default)]
    pub gy: f64,
    /// Gravity in z direction (m/s²). Default: -9.81.
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

/// Applies a constant gravitational body force to all local atoms.
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
            rotational_viscous: vec![],
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
            rotational_viscous: vec![],
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
            rotational_viscous: vec![],
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
            rotational_viscous: vec![],
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
            rotational_viscous: vec![],
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
            rotational_viscous: vec![],
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

    // ── Rotational viscous tests ───────────────────────────────────────────

    fn make_rotational_viscous_app(
        n: usize,
        group_name: &str,
        mask: Vec<bool>,
        gamma: f64,
        omegas: Vec<[f64; 3]>,
    ) -> App {
        let atoms = make_atoms(n);
        let groups = make_group_registry(group_name, mask);

        let mut dem = dem_atom::DemAtom::new();
        for omega in &omegas {
            dem.radius.push(0.5);
            dem.density.push(2500.0);
            dem.inv_inertia.push(1.0);
            dem.quaternion.push([1.0, 0.0, 0.0, 0.0]);
            dem.omega.push(*omega);
            dem.ang_mom.push([0.0; 3]);
            dem.torque.push([0.0; 3]);
        }

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let fixes = FixesRegistry {
            add_forces: vec![],
            set_forces: vec![],
            move_linears: vec![],
            freezes: vec![],
            viscous: vec![],
            rotational_viscous: vec![RotationalViscousDef {
                group: group_name.to_string(),
                gamma,
            }],
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_resource(fixes);
        app.add_update_system(apply_rotational_viscous, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();
        app
    }

    #[test]
    fn test_rotational_viscous_opposes_angular_velocity() {
        let app = make_rotational_viscous_app(
            2,
            "all",
            vec![true, true],
            0.1,
            vec![[1.0, -2.0, 0.5], [0.0; 3]],
        );

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<dem_atom::DemAtom>("test");
        // torque = -gamma * omega
        assert!((dem.torque[0][0] - (-0.1)).abs() < 1e-12, "tx = -gamma*wx");
        assert!((dem.torque[0][1] - 0.2).abs() < 1e-12, "ty = -gamma*wy");
        assert!(
            (dem.torque[0][2] - (-0.05)).abs() < 1e-12,
            "tz = -gamma*wz"
        );
    }

    #[test]
    fn test_rotational_viscous_zero_at_rest() {
        let app = make_rotational_viscous_app(
            2,
            "all",
            vec![true, true],
            0.1,
            vec![[0.0; 3], [0.0; 3]],
        );

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<dem_atom::DemAtom>("test");
        assert!((dem.torque[0][0]).abs() < 1e-15);
        assert!((dem.torque[0][1]).abs() < 1e-15);
        assert!((dem.torque[0][2]).abs() < 1e-15);
    }

    #[test]
    fn test_rotational_viscous_group_filtering() {
        let app = make_rotational_viscous_app(
            3,
            "subset",
            vec![true, false, true],
            0.5,
            vec![[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        );

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<dem_atom::DemAtom>("test");
        // Atom 0 in group: torque = -0.5 * 2.0 = -1.0
        assert!((dem.torque[0][0] - (-1.0)).abs() < 1e-12);
        // Atom 1 NOT in group: torque unchanged (0)
        assert!((dem.torque[1][0]).abs() < 1e-15);
        // Atom 2 in group: torque = -0.5 * 4.0 = -2.0
        assert!((dem.torque[2][0] - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_rotational_viscous_scales_with_gamma_and_omega() {
        // gamma=0.2, omega=[3.0, 0.0, 0.0] -> torque = -0.6
        let app = make_rotational_viscous_app(
            1,
            "all",
            vec![true],
            0.2,
            vec![[3.0, 0.0, 0.0]],
        );

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<dem_atom::DemAtom>("test");
        assert!(
            (dem.torque[0][0] - (-0.6)).abs() < 1e-12,
            "torque should scale linearly: -0.2*3.0 = -0.6, got {}",
            dem.torque[0][0]
        );
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
}

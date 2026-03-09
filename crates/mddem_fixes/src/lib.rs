//! Atom manipulation fixes: addforce, setforce, freeze, move_linear.

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, CommResource, Config, GroupRegistry};

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

// ── Registry ───────────────────────────────────────────────────────────────

pub struct FixesRegistry {
    pub add_forces: Vec<AddForceDef>,
    pub set_forces: Vec<SetForceDef>,
    pub move_linears: Vec<MoveLinearDef>,
    pub freezes: Vec<FreezeDef>,
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
# group = "frozen""#,
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
        };

        drop(config);

        let has_any = !registry.add_forces.is_empty()
            || !registry.set_forces.is_empty()
            || !registry.move_linears.is_empty()
            || !registry.freezes.is_empty();

        if !has_any {
            app.add_resource(registry);
            return;
        }

        let has_move = !registry.move_linears.is_empty();
        let has_add = !registry.add_forces.is_empty();
        let has_set = !registry.set_forces.is_empty();
        let has_freeze = !registry.freezes.is_empty();

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
}

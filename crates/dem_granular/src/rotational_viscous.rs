//! Rotational viscous damping: torque = -gamma * omega.

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use dem_atom::DemAtom;
use mddem_core::{Atom, AtomDataRegistry, CommResource, Config, GroupRegistry};

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct RotationalViscousDef {
    pub group: String,
    pub gamma: f64,
}

pub struct RotationalViscousRegistry {
    pub defs: Vec<RotationalViscousDef>,
}

pub struct RotationalViscousPlugin;

impl Plugin for RotationalViscousPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# [[rotational_viscous]]
# group = "all"
# gamma = 0.001  # angular damping coefficient (torque = -gamma * omega)"#,
        )
    }

    fn build(&self, app: &mut App) {
        let config = app
            .get_resource_ref::<Config>()
            .expect("Config resource must exist before RotationalViscousPlugin");

        let defs = config.parse_array::<RotationalViscousDef>("rotational_viscous");
        let has_any = !defs.is_empty();
        drop(config);

        let registry = RotationalViscousRegistry { defs };
        app.add_resource(registry);

        if has_any {
            app.add_setup_system(setup_rotational_viscous, ScheduleSetupSet::PostSetup);
            app.add_update_system(apply_rotational_viscous, ScheduleSet::PostForce);
        }
    }
}

fn setup_rotational_viscous(
    registry: Res<RotationalViscousRegistry>,
    comm: Res<CommResource>,
    groups: Res<GroupRegistry>,
) {
    for f in &registry.defs {
        groups.validate_name(&f.group, "fix rotational_viscous");
    }

    if comm.rank() != 0 {
        return;
    }
    for f in &registry.defs {
        println!(
            "Fix rotational_viscous: group='{}', gamma={}",
            f.group, f.gamma
        );
    }
}

/// Apply angular-velocity-proportional damping: τ = -gamma * ω.
fn apply_rotational_viscous(
    atoms: Res<Atom>,
    atom_data: Res<AtomDataRegistry>,
    registry: Res<RotationalViscousRegistry>,
    groups: Res<GroupRegistry>,
) {
    let mut dem = atom_data.expect_mut::<DemAtom>("apply_rotational_viscous");
    let nlocal = atoms.nlocal as usize;
    for def in &registry.defs {
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

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_test_utils::{make_atoms, make_group_registry};

    fn make_rotational_viscous_app(
        n: usize,
        group_name: &str,
        mask: Vec<bool>,
        gamma: f64,
        omegas: Vec<[f64; 3]>,
    ) -> App {
        let atoms = make_atoms(n);
        let groups = make_group_registry(group_name, mask);

        let mut dem = DemAtom::new();
        for omega in &omegas {
            dem.radius.push(0.5);
            dem.density.push(2500.0);
            dem.inv_inertia.push(1.0);
            dem.quaternion.push([1.0, 0.0, 0.0, 0.0]);
            dem.omega.push(*omega);
            dem.ang_mom.push([0.0; 3]);
            dem.torque.push([0.0; 3]);
        }

        let mut atom_registry = AtomDataRegistry::new();
        atom_registry.register(dem);

        let registry = RotationalViscousRegistry {
            defs: vec![RotationalViscousDef {
                group: group_name.to_string(),
                gamma,
            }],
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(atom_registry);
        app.add_resource(registry);
        app.add_update_system(apply_rotational_viscous, ScheduleSet::PostForce);
        app.organize_systems();
        app.run();
        app
    }

    #[test]
    fn test_opposes_angular_velocity() {
        let app = make_rotational_viscous_app(
            2,
            "all",
            vec![true, true],
            0.1,
            vec![[1.0, -2.0, 0.5], [0.0; 3]],
        );

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        assert!((dem.torque[0][0] - (-0.1)).abs() < 1e-12, "tx = -gamma*wx");
        assert!((dem.torque[0][1] - 0.2).abs() < 1e-12, "ty = -gamma*wy");
        assert!(
            (dem.torque[0][2] - (-0.05)).abs() < 1e-12,
            "tz = -gamma*wz"
        );
    }

    #[test]
    fn test_zero_at_rest() {
        let app = make_rotational_viscous_app(
            2,
            "all",
            vec![true, true],
            0.1,
            vec![[0.0; 3], [0.0; 3]],
        );

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        assert!((dem.torque[0][0]).abs() < 1e-15);
        assert!((dem.torque[0][1]).abs() < 1e-15);
        assert!((dem.torque[0][2]).abs() < 1e-15);
    }

    #[test]
    fn test_group_filtering() {
        let app = make_rotational_viscous_app(
            3,
            "subset",
            vec![true, false, true],
            0.5,
            vec![[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        );

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        assert!((dem.torque[0][0] - (-1.0)).abs() < 1e-12);
        assert!((dem.torque[1][0]).abs() < 1e-15);
        assert!((dem.torque[2][0] - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_scales_with_gamma_and_omega() {
        let app = make_rotational_viscous_app(
            1,
            "all",
            vec![true],
            0.2,
            vec![[3.0, 0.0, 0.0]],
        );

        let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
        let dem = registry.expect::<DemAtom>("test");
        assert!(
            (dem.torque[0][0] - (-0.6)).abs() < 1e-12,
            "torque should scale linearly: -0.2*3.0 = -0.6, got {}",
            dem.torque[0][0]
        );
    }
}

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;

use dem_atom::{DemAtom, MaterialTable};
use mddem_core::{Atom, AtomDataRegistry};
use mddem_neighbor::Neighbor;

// √(5/3) — appears in the viscoelastic damping formula
const SQRT_5_3: f64 = 0.9128709291752768;

/// Hertz elastic contact force with viscoelastic normal damping.
pub struct HertzNormalForcePlugin;

impl Plugin for HertzNormalForcePlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(hertz_normal_force.label("hertz_normal"), ScheduleSet::Force);
    }
}

pub fn hertz_normal_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
) {
    let dem = registry.expect::<DemAtom>("hertz_normal_force");

    let nlocal = atoms.nlocal as usize;
    let mut overlap_warnings = 0usize;
    for (i, j) in neighbor.pairs(nlocal) {

        let p1 = Vector3::new(atoms.pos_x[i], atoms.pos_y[i], atoms.pos_z[i]);
        let p2 = Vector3::new(atoms.pos_x[j], atoms.pos_y[j], atoms.pos_z[j]);

        let r1 = dem.radius[i];
        let r2 = dem.radius[j];

        let diff = p2 - p1;
        let distance = diff.norm();

        if distance >= r1 + r2 {
            continue;
        }

        if distance == 0.0 {
            eprintln!(
                "WARNING: zero separation between tags {} {}",
                atoms.tag[i], atoms.tag[j]
            );
            continue;
        }

        if distance / (r1 + r2) < 0.90 {
            overlap_warnings += 1;
            eprintln!(
                "WARNING: large overlap tags {} {} ratio {:.3}",
                atoms.tag[i],
                atoms.tag[j],
                distance / (r1 + r2)
            );
            if overlap_warnings > 100 {
                panic!(
                    "Over 100 excessive overlaps this step — aborting. \
                     Check timestep or initial configuration."
                );
            }
            continue;
        }

        atoms.is_collision[i] = true;
        atoms.is_collision[j] = true;

        let n = diff / distance;
        let delta = (r1 + r2) - distance;

        let mat_i = atoms.atom_type[i] as usize;
        let mat_j = atoms.atom_type[j] as usize;

        let r_eff = (r1 * r2) / (r1 + r2);
        let e_eff = 1.0
            / ((1.0 - material_table.poisson_ratio[mat_i].powi(2))
                / material_table.youngs_mod[mat_i]
                + (1.0 - material_table.poisson_ratio[mat_j].powi(2))
                    / material_table.youngs_mod[mat_j]);

        let sqrt_dr = (delta * r_eff).sqrt();
        let s_n = 2.0 * e_eff * sqrt_dr;
        let k_n = 4.0 / 3.0 * e_eff * sqrt_dr;

        let m_r = (atoms.mass[i] * atoms.mass[j]) / (atoms.mass[i] + atoms.mass[j]);

        let v_rel = Vector3::new(
            atoms.vel_x[j] - atoms.vel_x[i],
            atoms.vel_y[j] - atoms.vel_y[i],
            atoms.vel_z[j] - atoms.vel_z[i],
        );
        let v_n = v_rel.dot(&n);

        let beta = material_table.beta_ij[mat_i][mat_j];

        let f_spring = k_n * delta;
        let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
        let f_net = (f_spring - f_diss).max(0.0);

        let force = f_net * n;

        atoms.force_x[i] -= force.x;
        atoms.force_y[i] -= force.y;
        atoms.force_z[i] -= force.z;
        atoms.force_x[j] += force.x;
        atoms.force_y[j] += force.y;
        atoms.force_z[j] += force.z;

    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use dem_atom::{DemAtom, MaterialTable};
    use mddem_core::{Atom, AtomDataRegistry};
    use mddem_neighbor::Neighbor;
    use nalgebra::Vector3;

    fn push_test_atom(
        atom: &mut Atom,
        dem: &mut DemAtom,
        tag: u32,
        pos: Vector3<f64>,
        radius: f64,
    ) {
        let mass = 2500.0 * 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
        atom.push_test_atom(tag, pos, radius, mass);
        dem.radius.push(radius);
        dem.density.push(2500.0);
    }

    fn make_material_table() -> MaterialTable {
        let mut mt = MaterialTable::new();
        mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4);
        mt.build_pair_tables();
        mt
    }

    #[test]
    fn hertz_repulsive_for_overlap() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();

        let radius = 0.001;
        push_test_atom(&mut atom, &mut dem, 0, Vector3::new(0.0, 0.0, 0.0), radius);
        push_test_atom(
            &mut atom,
            &mut dem,
            1,
            Vector3::new(0.0019, 0.0, 0.0),
            radius,
        );
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_offsets = vec![0, 1, 1];
        neighbor.neighbor_indices = vec![1];

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_update_system(hertz_normal_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            atom.force_x[0] < 0.0,
            "particle 0 should have negative x force"
        );
        assert!(
            atom.force_x[1] > 0.0,
            "particle 1 should have positive x force"
        );
        assert!((atom.force_x[0] + atom.force_x[1]).abs() < 1e-10);
    }

    #[test]
    fn hertz_zero_for_gap() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();

        let radius = 0.001;
        push_test_atom(&mut atom, &mut dem, 0, Vector3::new(0.0, 0.0, 0.0), radius);
        push_test_atom(
            &mut atom,
            &mut dem,
            1,
            Vector3::new(0.003, 0.0, 0.0),
            radius,
        );
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.neighbor_offsets = vec![0, 1, 1];
        neighbor.neighbor_indices = vec![1];

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_update_system(hertz_normal_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!((atom.force_x[0]).abs() < 1e-20);
        assert!((atom.force_x[1]).abs() < 1e-20);
    }
}

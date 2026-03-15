use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use dem_atom::{DemAtom, MaterialTable};
use mddem_core::{Atom, AtomDataRegistry, BondStore};
use mddem_neighbor::Neighbor;

use crate::{LARGE_OVERLAP_WARN_THRESHOLD, MAX_OVERLAP_WARNINGS, SQRT_5_3};

/// Hertz elastic contact force with viscoelastic normal damping.
pub struct HertzNormalForcePlugin;

impl Plugin for HertzNormalForcePlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(hertz_normal_force, ScheduleSet::Force);
    }
}

pub fn hertz_normal_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
) {
    let dem = registry.expect::<DemAtom>("hertz_normal_force");
    let bond_store = registry.get::<BondStore>();

    let nlocal = atoms.nlocal as usize;
    let mut overlap_warnings = 0usize;
    for (i, j) in neighbor.pairs(nlocal) {
        if let Some(ref bonds) = bond_store {
            if bonds.are_excluded(i, j, &atoms.tag) {
                continue;
            }
        }


        let r1 = dem.radius[i];
        let r2 = dem.radius[j];

        let dx = atoms.pos[j][0] - atoms.pos[i][0];
        let dy = atoms.pos[j][1] - atoms.pos[i][1];
        let dz = atoms.pos[j][2] - atoms.pos[i][2];
        let distance = (dx*dx + dy*dy + dz*dz).sqrt();

        if distance >= r1 + r2 {
            continue;
        }

        if distance == 0.0 {
            #[cfg(debug_assertions)]
            eprintln!(
                "WARNING: zero separation between tags {} {}",
                atoms.tag[i], atoms.tag[j]
            );
            continue;
        }

        if distance / (r1 + r2) < LARGE_OVERLAP_WARN_THRESHOLD {
            overlap_warnings += 1;
            #[cfg(debug_assertions)]
            eprintln!(
                "WARNING: large overlap tags {} {} ratio {:.3}",
                atoms.tag[i],
                atoms.tag[j],
                distance / (r1 + r2)
            );
            if overlap_warnings > MAX_OVERLAP_WARNINGS {
                panic!(
                    "Over {} excessive overlaps this step — aborting. \
                     Check timestep or initial configuration.",
                    MAX_OVERLAP_WARNINGS
                );
            }
            continue;
        }

        let inv_dist = 1.0 / distance;
        let nx = dx * inv_dist;
        let ny = dy * inv_dist;
        let nz = dz * inv_dist;
        let delta = (r1 + r2) - distance;

        let mat_i = atoms.atom_type[i] as usize;
        let mat_j = atoms.atom_type[j] as usize;

        let r_eff = (r1 * r2) / (r1 + r2);
        let e_eff = material_table.e_eff_ij[mat_i][mat_j];

        let sqrt_dr = (delta * r_eff).sqrt();
        let s_n = 2.0 * e_eff * sqrt_dr;
        let k_n = 4.0 / 3.0 * e_eff * sqrt_dr;

        let m_r = 1.0 / (atoms.inv_mass[i] + atoms.inv_mass[j]);

        let vrx = atoms.vel[j][0] - atoms.vel[i][0];
        let vry = atoms.vel[j][1] - atoms.vel[i][1];
        let vrz = atoms.vel[j][2] - atoms.vel[i][2];
        let v_n = vrx*nx + vry*ny + vrz*nz;

        let beta = material_table.beta_ij[mat_i][mat_j];

        let f_spring = k_n * delta;
        let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
        let f_net = (f_spring - f_diss).max(0.0);

        let fx = f_net * nx;
        let fy = f_net * ny;
        let fz = f_net * nz;

        atoms.force[i][0] -= fx;
        atoms.force[i][1] -= fy;
        atoms.force[i][2] -= fz;
        atoms.force[j][0] += fx;
        atoms.force[j][1] += fy;
        atoms.force[j][2] += fz;

    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use dem_atom::DemAtom;
    use mddem_core::{Atom, AtomDataRegistry};
    use mddem_neighbor::Neighbor;
    use mddem_test_utils::{make_material_table, push_dem_test_atom};

    #[test]
    fn hertz_repulsive_for_overlap() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();

        let radius = 0.001;
        push_dem_test_atom(&mut atom, &mut dem, 0, [0.0, 0.0, 0.0], radius);
        push_dem_test_atom(
            &mut atom,
            &mut dem,
            1,
            [0.0019, 0.0, 0.0],
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
            atom.force[0][0] < 0.0,
            "particle 0 should have negative x force"
        );
        assert!(
            atom.force[1][0] > 0.0,
            "particle 1 should have positive x force"
        );
        assert!((atom.force[0][0] + atom.force[1][0]).abs() < 1e-10);
    }

    #[test]
    fn hertz_zero_for_gap() {
        let mut app = App::new();

        let mut atom = Atom::new();
        let mut dem = DemAtom::new();

        let radius = 0.001;
        push_dem_test_atom(&mut atom, &mut dem, 0, [0.0, 0.0, 0.0], radius);
        push_dem_test_atom(
            &mut atom,
            &mut dem,
            1,
            [0.003, 0.0, 0.0],
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
        assert!((atom.force[0][0]).abs() < 1e-20);
        assert!((atom.force[1][0]).abs() < 1e-20);
    }
}

//! Standalone Hertz normal contact force (without tangential friction).
//!
//! This module provides a normal-only Hertz contact force for cases where
//! tangential friction is not needed. For full contact physics including
//! tangential, rolling, and twisting, use [`crate::contact::HertzMindlinContactPlugin`].
//!
//! # Physics
//!
//! The Hertz contact force for two elastic spheres with overlap `δ`:
//!
//! - Elastic stiffness: `k_n = 4/3 E* √(R* δ)`
//! - Spring force: `F_spring = k_n δ`
//! - Damping force: `F_damp = 2 β √(5/3) √(S_n m_r) v_n`
//! - Net force: `F_n = max(F_spring - F_damp, 0)` (no tensile normal force)
//!
//! where `E*` is the effective Young's modulus, `R*` is the effective radius,
//! `β` is the damping coefficient (derived from restitution), `S_n = 2 E* √(R* δ)`,
//! and `m_r` is the reduced mass.

use sim_app::prelude::*;
use sim_scheduler::prelude::*;

use dem_atom::{DemAtom, MaterialTable};
use mddem_core::{Atom, AtomDataRegistry, BondStore, ParticleSimScheduleSet};
use mddem_core::Neighbor;

use crate::{LARGE_OVERLAP_WARN_THRESHOLD, MAX_OVERLAP_WARNINGS, SQRT_5_3};

/// Standalone Hertz elastic normal contact force plugin.
///
/// Registers the [`hertz_normal_force`] system at [`ParticleSimScheduleSet::Force`].
/// Does **not** include tangential friction, rolling, or twisting — for full
/// contact physics use [`crate::contact::HertzMindlinContactPlugin`].
pub struct HertzNormalForcePlugin;

impl Plugin for HertzNormalForcePlugin {
    fn build(&self, app: &mut App) {
        app.add_update_system(hertz_normal_force, ParticleSimScheduleSet::Force);
    }
}

/// Compute Hertz elastic normal contact forces for all neighbor pairs.
///
/// For each overlapping pair, computes the repulsive normal force from Hertz
/// contact theory with viscoelastic damping. Forces are accumulated into
/// `atoms.force` with Newton's third law (equal and opposite).
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

        // Effective radius: R* = R1 R2 / (R1 + R2)
        let r_eff = (r1 * r2) / (r1 + r2);
        // Effective Young's modulus: 1/E* = (1-ν1²)/E1 + (1-ν2²)/E2
        let e_eff = material_table.e_eff_ij[mat_i][mat_j];

        // √(δ R*) appears in both stiffness and damping terms
        let sqrt_dr = (delta * r_eff).sqrt();
        // Normal stiffness parameter: S_n = 2 E* √(δ R*)
        let s_n = 2.0 * e_eff * sqrt_dr;
        // Hertz spring constant: k_n = 4/3 E* √(δ R*)
        let k_n = 4.0 / 3.0 * e_eff * sqrt_dr;

        // Reduced mass: m_r = m1 m2 / (m1 + m2) = 1 / (1/m1 + 1/m2)
        let m_r = 1.0 / (atoms.inv_mass[i] + atoms.inv_mass[j]);

        // Relative velocity (j relative to i) projected onto contact normal
        let vrx = atoms.vel[j][0] - atoms.vel[i][0];
        let vry = atoms.vel[j][1] - atoms.vel[i][1];
        let vrz = atoms.vel[j][2] - atoms.vel[i][2];
        let v_n = vrx*nx + vry*ny + vrz*nz;

        // Damping coefficient: β = ln(e) / √(ln²(e) + π²) where e = restitution
        let beta = material_table.beta_ij[mat_i][mat_j];

        // F_spring = k_n δ (repulsive, proportional to overlap)
        let f_spring = k_n * delta;
        // F_damp = 2 β √(5/3) √(S_n m_r) v_n (dissipative, proportional to approach velocity)
        let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
        // Net force clamped to ≥ 0 (no tensile normal force without adhesion)
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
    use mddem_core::Neighbor;
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
        app.add_update_system(hertz_normal_force, ParticleSimScheduleSet::Force);
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
        app.add_update_system(hertz_normal_force, ParticleSimScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!((atom.force[0][0]).abs() < 1e-20);
        assert!((atom.force[1][0]).abs() < 1e-20);
    }
}

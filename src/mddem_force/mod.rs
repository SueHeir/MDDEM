use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;

use crate::{
    dem_atom::DemAtom,
    mddem_atom::{Atom, AtomDataRegistry},
    mddem_communication::Comm,
    mddem_neighbor::Neighbor,
};

pub struct ForcePlugin;

impl Plugin for ForcePlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Force::new())
            .add_update_system(hertz_normal_force, ScheduleSet::Force);
    }
}

pub struct Force {
    pub _skin_fraction: f64,
}

impl Force {
    pub fn new() -> Self {
        Force { _skin_fraction: 1.0 }
    }
}


pub fn hertz_normal_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    comm: Res<Comm>,
) {
    // Borrow DemAtom once outside the loop — no borrow conflict with atoms.force
    // because registry and atoms are separate ECS resources.
    let dem = registry.get::<DemAtom>().unwrap();

    for &(i, j) in &neighbor.neighbor_list {
        let p1 = atoms.pos[i];
        let p2 = atoms.pos[j];
        let v1 = atoms.velocity[i];
        let v2 = atoms.velocity[j];

        let r1 = dem.radius[i];
        let r2 = dem.radius[j];
        let beta = dem.beta;
        let poisson_ratio1 = dem.poisson_ratio[i];
        let poisson_ratio2 = dem.poisson_ratio[j];
        let youngs_mod1 = dem.youngs_mod[i];
        let youngs_mod2 = dem.youngs_mod[j];

        let position_difference = p2 - p1;
        let distance = position_difference.norm();

        if distance == 0.0 {
            println!("tags {} {}", atoms.tag[i], atoms.tag[j]);
        }

        if distance < r1 + r2 {
            if distance / (r1 + r2) < 0.90 {
                println!("large overlap");
                println!("isGhost: {} {}", atoms.is_ghost[i], atoms.is_ghost[j]);
                println!("hasGhost: {} {}", atoms.has_ghost[i], atoms.has_ghost[j]);
                println!("pos: {} {}", atoms.pos[i], atoms.pos[j]);
                panic!()
            }

            atoms.is_collision[i] = true;
            atoms.is_collision[j] = true;

            let normalized_delta = position_difference / distance;
            let distance_delta = (r1 + r2) - distance;

            let effective_radius = (r1 * r2) / (r1 + r2);

            let effective_youngs = 1.0
                / ((1.0 - poisson_ratio1 * poisson_ratio1) / youngs_mod1
                    + (1.0 - poisson_ratio2 * poisson_ratio2) / youngs_mod2);

            let g_effective = 1.0 / (
                2.0 * (2.0 - poisson_ratio1) * (1.0 + poisson_ratio1) / youngs_mod1
                + 2.0 * (2.0 - poisson_ratio2) * (1.0 + poisson_ratio2) / youngs_mod2
            );

            let sqrtvalue = (distance_delta * effective_radius).sqrt();

            let reduced_mass = (atoms.mass[i] * atoms.mass[j]) / (atoms.mass[i] + atoms.mass[j]);

            let sn = 2.0 * effective_youngs * sqrtvalue;
            let st = 8.0 * g_effective * sqrtvalue;
            let kn = 4.0 / 3.0 * effective_youngs * sqrtvalue;

            let gammat = -2.0 * 0.91287092917527685576161630466800355658790782499663875
                * beta * (st * reduced_mass).sqrt();

            let normal_force = kn * distance_delta;

            let delta_velocity = v2 - v1;
            let v_r_n = delta_velocity.dot(&normalized_delta);
            let vrn = v_r_n * normalized_delta;
            let vrt = delta_velocity - vrn;
            let vrel = vrt.norm();

            // Coulomb friction cap
            let ft_friction = 0.4 * normal_force.abs();
            let gamma = if gammat.abs() * vrel > ft_friction {
                -ft_friction / vrel
            } else {
                gammat
            };

            let ft = -gamma * vrt;

            let dissipation_force = 2.0
                * beta
                * 0.91287092917527685576161630466800355658790782499663875
                * (sn * reduced_mass).sqrt()
                * v_r_n;

            // No-tension clamp
            let net_normal = (normal_force - dissipation_force).max(0.0);

            if atoms.is_ghost[i] || atoms.is_ghost[j] {
                // Cross-boundary pair: computed on both processors, so halve to avoid double-counting.
                atoms.force[i] -= (net_normal * normalized_delta - ft) * 0.5;
                atoms.force[j] += (net_normal * normalized_delta - ft) * 0.5;
            } else {
                atoms.force[i] -= net_normal * normalized_delta - ft;
                atoms.force[j] += net_normal * normalized_delta - ft;
            }
        }
    }
}

//! Bonded chain pull test: a straight chain of particles along the z-axis,
//! bottom particle frozen, top particle pulled upward by a constant force.
//!
//! Demonstrates auto-bonding, bond normal forces, and force/freeze fixes.
//!
//! ```bash
//! cargo run --example bond_basic --no-default-features -- examples/bond_basic/config.toml
//! ```

use std::f64::consts::PI;

use mddem::dem_atom::DemAtom;
use mddem::prelude::*;
use nalgebra::UnitQuaternion;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(DemBondPlugin)
        .add_plugins(FixesPlugin);
    app.add_setup_system(setup_chain, ScheduleSetupSet::Setup);
    app.start();
}

/// Place 10 particles in a straight line along the z-axis, just touching.
fn setup_chain(
    mut atom: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
    scheduler_manager: Res<SchedulerManager>,
) {
    if scheduler_manager.index != 0 {
        return;
    }

    let n_particles: usize = 10;
    let radius: f64 = 0.001;
    let density: f64 = 2500.0;
    // Particles just touching (distance = 2*radius between centers)
    let spacing = 2.0 * radius;
    let mat_idx = material_table
        .find_material("glass")
        .expect("material 'glass' not found");

    let mut dem = registry.expect_mut::<DemAtom>("setup_chain");

    for i in 0..n_particles {
        let z = radius + (i as f64) * spacing;
        let mass = density * (4.0 / 3.0) * PI * radius.powi(3);
        let tag = atom.get_max_tag() + 1;

        atom.natoms += 1;
        atom.nlocal += 1;
        atom.tag.push(tag);
        atom.origin_index.push(0);
        atom.skin.push(radius);
        atom.is_collision.push(false);
        atom.is_ghost.push(false);
        atom.pos.push([0.0, 0.0, z]);
        atom.vel.push([0.0; 3]);
        atom.force.push([0.0; 3]);
        atom.mass.push(mass);
        atom.inv_mass.push(1.0 / mass);
        atom.atom_type.push(mat_idx);

        dem.radius.push(radius);
        dem.density.push(density);
        dem.inv_inertia.push(1.0 / (0.4 * mass * radius * radius));
        dem.quaternion.push(UnitQuaternion::identity());
        dem.omega.push([0.0; 3]);
        dem.ang_mom.push([0.0; 3]);
        dem.torque.push([0.0; 3]);
    }
}

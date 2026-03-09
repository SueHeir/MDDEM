//! Random particle insertion for DEM simulations from `[[particles.insert]]` config blocks.

use std::f64::consts::PI;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::UnitQuaternion;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;

use dem_atom::{DemAtom, MaterialTable};
use mddem_core::{Atom, AtomDataRegistry, CommResource, Config, Domain};

// ── Config structs ──────────────────────────────────────────────────────────

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// A single particle insertion block from `[[particles.insert]]`.
pub struct InsertConfig {
    pub material: String,
    pub count: u32,
    pub radius: f64,
    pub density: f64,
    pub velocity: Option<f64>,
    pub velocity_x: Option<f64>,
    pub velocity_y: Option<f64>,
    pub velocity_z: Option<f64>,
    pub region_x_low: Option<f64>,
    pub region_x_high: Option<f64>,
    pub region_y_low: Option<f64>,
    pub region_y_high: Option<f64>,
    pub region_z_low: Option<f64>,
    pub region_z_high: Option<f64>,
}

/// TOML `[particles]` — contains a list of insertion blocks.
#[derive(Deserialize, Clone, Default)]
pub struct ParticlesConfig {
    pub insert: Option<Vec<InsertConfig>>,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Inserts DEM particles at setup time based on `[[particles.insert]]` config blocks.
pub struct DemAtomInsertPlugin;

impl Plugin for DemAtomInsertPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# Particle insertion blocks (one per material/group)
[[particles.insert]]
material = "glass"          # must match a [[dem.materials]] name
count = 100
radius = 0.001
density = 2500.0
# velocity = 0.1            # random velocity magnitude (Gaussian)
# velocity_x = 0.0          # directional velocity (additive with random)
# velocity_y = 0.0
# velocity_z = 0.0
# region_x_low = 0.0        # insertion region (defaults to domain bounds)
# region_x_high = 1.0
# region_y_low = 0.0
# region_y_high = 1.0
# region_z_low = 0.0
# region_z_high = 1.0"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<ParticlesConfig>(app, "particles");

        app.add_setup_system(dem_insert_atoms, ScheduleSetupSet::Setup)
            .add_setup_system(calculate_delta_time, ScheduleSetupSet::PostSetup);
    }
}

// ── Insertion system ────────────────────────────────────────────────────────

pub fn dem_insert_atoms(
    particles_config: Res<ParticlesConfig>,
    comm: Res<CommResource>,
    domain: Res<Domain>,
    mut atom: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
    scheduler_manager: Res<SchedulerManager>,
) {
    // Only insert particles on the first stage
    if scheduler_manager.index != 0 {
        return;
    }

    // Insert particles per insert block
    if let Some(ref inserts) = particles_config.insert {
        if comm.rank() == 0 {
            let mut dem_data = registry.expect_mut::<DemAtom>("dem_insert_atoms");
            let mut rng = rand::rng();
            let mut max_tag = atom.get_max_tag();

            for insert in inserts {
                // Find material config by name
                let mat_idx = match material_table.find_material(&insert.material) {
                    Some(idx) => idx,
                    None => {
                        eprintln!(
                            "ERROR: Unknown material '{}' in [[particles.insert]]. Available: {:?}",
                            insert.material, material_table.names
                        );
                        std::process::exit(1);
                    }
                };

                let radius = insert.radius;
                let density = insert.density;

                println!("DemAtomInsert: inserting {} particles of material '{}' (r={}, rho={}, E={}, nu={})",
                    insert.count, insert.material, radius, density,
                    material_table.youngs_mod[mat_idx as usize],
                    material_table.poisson_ratio[mat_idx as usize]);

                let x_lo = insert.region_x_low.unwrap_or(domain.boundaries_low[0]) + radius;
                let x_hi = insert.region_x_high.unwrap_or(domain.boundaries_high[0]) - radius;
                let y_lo = insert.region_y_low.unwrap_or(domain.boundaries_low[1]) + radius;
                let y_hi = insert.region_y_high.unwrap_or(domain.boundaries_high[1]) - radius;
                let z_lo = insert.region_z_low.unwrap_or(domain.boundaries_low[2]) + radius;
                let z_hi = insert.region_z_high.unwrap_or(domain.boundaries_high[2]) - radius;

                let mut count = 0u32;
                while count < insert.count {
                    let x = rng.random_range(x_lo..x_hi);
                    let y = rng.random_range(y_lo..y_hi);
                    let z = rng.random_range(z_lo..z_hi);

                    let mut no_overlap = true;
                    for i in 0..atom.len() {
                        let dx = x - atom.pos[i][0];
                        let dy = y - atom.pos[i][1];
                        let dz = z - atom.pos[i][2];
                        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                        if distance <= (radius + dem_data.radius[i]) * 1.1 {
                            no_overlap = false;
                            break;
                        }
                    }

                    if no_overlap {
                        count += 1;
                        atom.natoms += 1;
                        atom.nlocal += 1;
                        atom.tag.push(max_tag);
                        atom.origin_index.push(0);
                        atom.skin.push(radius);
                        atom.is_collision.push(false);
                        atom.is_ghost.push(false);
                                                max_tag += 1;
                        atom.pos.push([x, y, z]);
                        atom.vel.push([0.0; 3]);
                        atom.quaternion.push(UnitQuaternion::identity());
                        atom.omega.push([0.0; 3]);
                        atom.ang_mom.push([0.0; 3]);
                        atom.torque.push([0.0; 3]);
                        atom.force.push([0.0; 3]);
                        let mass = density * 4.0 / 3.0 * PI * radius.powi(3);
                        atom.mass.push(mass);
                        atom.inv_mass.push(1.0 / mass);

                        atom.atom_type.push(mat_idx);
                        dem_data.radius.push(radius);
                        dem_data.density.push(density);
                        dem_data.inv_inertia.push(1.0 / (0.4 * mass * radius * radius));
                    }
                }

                // Apply per-insert velocity to this batch
                let total_len = atom.vel.len();
                let start = total_len - insert.count as usize;
                if let Some(rand_vel) = insert.velocity {
                    if rand_vel < 0.0 {
                        eprintln!("ERROR: velocity in [[particles.insert]] must be non-negative, got {}", rand_vel);
                        std::process::exit(1);
                    }
                    let normal = Normal::new(0.0, rand_vel).unwrap();
                    for i in start..total_len {
                        atom.vel[i][0] = normal.sample(&mut rng);
                        atom.vel[i][1] = normal.sample(&mut rng);
                        atom.vel[i][2] = normal.sample(&mut rng);
                    }
                }
                // Apply directional velocity components (additive with random)
                let vx = insert.velocity_x.unwrap_or(0.0);
                let vy = insert.velocity_y.unwrap_or(0.0);
                let vz = insert.velocity_z.unwrap_or(0.0);
                if vx != 0.0 || vy != 0.0 || vz != 0.0 {
                    for i in start..total_len {
                        atom.vel[i][0] += vx;
                        atom.vel[i][1] += vy;
                        atom.vel[i][2] += vz;
                    }
                }
            }
        }
    }
}

fn calculate_delta_time(
    comm: Res<CommResource>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
    scheduler_manager: Res<SchedulerManager>,
) {
    if scheduler_manager.index != 0 {
        return;
    }
    let dem = registry.expect::<DemAtom>("calculate_delta_time");
    let mut dt: f64 = 0.001;

    for i in 0..dem.radius.len() {
        let mat_idx = atoms.atom_type[i] as usize;
        let youngs_mod = material_table.youngs_mod[mat_idx];
        let poisson_ratio = material_table.poisson_ratio[mat_idx];
        let g = youngs_mod / (2.0 * (1.0 + poisson_ratio));
        let alpha = 0.1631 * poisson_ratio + 0.876605;
        let delta = PI * dem.radius[i] / alpha * (dem.density[i] / g).sqrt();
        dt = delta.min(dt);
    }

    dt = comm.all_reduce_min_f64(dt);

    if comm.rank() == 0 {
        println!("Using {} for delta time", dt * 0.15);
    }
    atoms.dt = dt * 0.15;
}

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use mddem_core::{Atom, AtomDataRegistry, Config};
use dem_atom::{DemAtom, MaterialTable};

// √(5/3) — appears in the viscoelastic damping formula
const SQRT_5_3: f64 = 0.9128709291752768;

fn default_neg_inf() -> f64 { f64::NEG_INFINITY }
fn default_pos_inf() -> f64 { f64::INFINITY }

#[derive(Deserialize, Clone)]
pub struct WallDef {
    pub point_x: f64,
    pub point_y: f64,
    pub point_z: f64,
    pub normal_x: f64,
    pub normal_y: f64,
    pub normal_z: f64,
    pub material: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "default_neg_inf")]
    pub bound_x_low: f64,
    #[serde(default = "default_pos_inf")]
    pub bound_x_high: f64,
    #[serde(default = "default_neg_inf")]
    pub bound_y_low: f64,
    #[serde(default = "default_pos_inf")]
    pub bound_y_high: f64,
    #[serde(default = "default_neg_inf")]
    pub bound_z_low: f64,
    #[serde(default = "default_pos_inf")]
    pub bound_z_high: f64,
}

pub struct WallPlane {
    pub point_x: f64,
    pub point_y: f64,
    pub point_z: f64,
    pub normal_x: f64,
    pub normal_y: f64,
    pub normal_z: f64,
    pub material_index: usize,
    pub name: Option<String>,
    pub bound_x_low: f64,
    pub bound_x_high: f64,
    pub bound_y_low: f64,
    pub bound_y_high: f64,
    pub bound_z_low: f64,
    pub bound_z_high: f64,
}

impl WallPlane {
    /// Check if atom position is within the wall's bounding region.
    /// Uses the atom position directly (not the projected point on the plane).
    #[inline]
    fn in_bounds(&self, x: f64, y: f64, z: f64) -> bool {
        x >= self.bound_x_low && x <= self.bound_x_high
            && y >= self.bound_y_low && y <= self.bound_y_high
            && z >= self.bound_z_low && z <= self.bound_z_high
    }
}

pub struct Walls {
    pub planes: Vec<WallPlane>,
    pub active: Vec<bool>,
}

impl Walls {
    pub fn deactivate_by_name(&mut self, name: &str) {
        for (i, wall) in self.planes.iter().enumerate() {
            if wall.name.as_deref() == Some(name) {
                self.active[i] = false;
            }
        }
    }
}

pub struct WallPlugin;

impl Plugin for WallPlugin {
    fn build(&self, app: &mut App) {
        let walls = {
            let config = app.get_resource_ref::<Config>().expect("Config resource must exist");
            let wall_defs: Vec<WallDef> = if let Some(val) = config.table.get("wall") {
                match val {
                    toml::Value::Array(arr) => {
                        arr.iter()
                            .map(|v| v.clone().try_into::<WallDef>().expect("failed to parse [[wall]] entry"))
                            .collect()
                    }
                    toml::Value::Table(t) => {
                        vec![toml::Value::Table(t.clone()).try_into::<WallDef>().expect("failed to parse [wall] entry")]
                    }
                    _ => panic!("[wall] must be a table or array of tables"),
                }
            } else {
                Vec::new()
            };
            drop(config);

            let material_table = app.get_resource_ref::<MaterialTable>()
                .expect("MaterialTable must exist before WallPlugin — add DemAtomPlugin first");

            let mut planes = Vec::with_capacity(wall_defs.len());
            for w in &wall_defs {
                let mat_idx = material_table.find_material(&w.material)
                    .unwrap_or_else(|| panic!("wall material '{}' not found in [[dem.materials]]", w.material)) as usize;
                let mag = (w.normal_x * w.normal_x + w.normal_y * w.normal_y + w.normal_z * w.normal_z).sqrt();
                assert!(mag > 1e-15, "wall normal vector must be non-zero");
                planes.push(WallPlane {
                    point_x: w.point_x,
                    point_y: w.point_y,
                    point_z: w.point_z,
                    normal_x: w.normal_x / mag,
                    normal_y: w.normal_y / mag,
                    normal_z: w.normal_z / mag,
                    material_index: mat_idx,
                    name: w.name.clone(),
                    bound_x_low: w.bound_x_low,
                    bound_x_high: w.bound_x_high,
                    bound_y_low: w.bound_y_low,
                    bound_y_high: w.bound_y_high,
                    bound_z_low: w.bound_z_low,
                    bound_z_high: w.bound_z_high,
                });
            }
            drop(material_table);

            let n = planes.len();
            Walls {
                planes,
                active: vec![true; n],
            }
        };

        app.add_resource(walls);
        app.add_update_system(
            wall_contact_force.label("wall_contact"),
            ScheduleSet::Force,
        );
    }
}

pub fn wall_contact_force(
    mut atoms: ResMut<Atom>,
    walls: Res<Walls>,
    registry: Res<AtomDataRegistry>,
    material_table: Res<MaterialTable>,
) {
    let dem = registry.get::<DemAtom>().unwrap();

    for (wall_idx, wall) in walls.planes.iter().enumerate() {
        if !walls.active[wall_idx] {
            continue;
        }

        let wall_mat = wall.material_index;

        for i in 0..atoms.nlocal as usize {
            let px = atoms.pos_x[i];
            let py = atoms.pos_y[i];
            let pz = atoms.pos_z[i];

            // Check if atom is within the wall's bounding region
            if !wall.in_bounds(px, py, pz) {
                continue;
            }

            // Vector from wall point to atom position
            let dx = px - wall.point_x;
            let dy = py - wall.point_y;
            let dz = pz - wall.point_z;

            // Signed distance from atom to wall plane (positive = on normal side)
            let distance = dx * wall.normal_x + dy * wall.normal_y + dz * wall.normal_z;

            // Only apply force when atom center is on the normal side of the wall
            if distance <= 0.0 {
                continue;
            }

            let radius = dem.radius[i];
            let delta = (radius - distance).min(0.5 * radius);

            if delta <= 0.0 {
                continue;
            }

            let mat_i = atoms.atom_type[i] as usize;

            // Wall has infinite radius → r_eff = r_particle
            let r_eff = radius;
            let e_eff = 1.0
                / ((1.0 - material_table.poisson_ratio[mat_i].powi(2)) / material_table.youngs_mod[mat_i]
                    + (1.0 - material_table.poisson_ratio[wall_mat].powi(2)) / material_table.youngs_mod[wall_mat]);

            let sqrt_dr = (delta * r_eff).sqrt();
            let s_n = 2.0 * e_eff * sqrt_dr;
            let k_n = 4.0 / 3.0 * e_eff * sqrt_dr;

            // Wall has infinite mass → m_reduced = m_particle
            let m_r = atoms.mass[i];

            // Relative velocity along wall normal: positive = separating, negative = approaching
            // Matches particle-particle convention where v_n = v_rel . n with n pointing from atom toward wall
            let v_n = atoms.vel_x[i] * wall.normal_x
                    + atoms.vel_y[i] * wall.normal_y
                    + atoms.vel_z[i] * wall.normal_z;

            let beta = material_table.beta_ij[mat_i][wall_mat];

            let f_spring = k_n * delta;
            let f_diss = 2.0 * beta * SQRT_5_3 * (s_n * m_r).sqrt() * v_n;
            let f_net = (f_spring - f_diss).max(0.0);

            // Force direction: along wall normal (pushes atom away from wall)
            atoms.force_x[i] += f_net * wall.normal_x;
            atoms.force_y[i] += f_net * wall.normal_y;
            atoms.force_z[i] += f_net * wall.normal_z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::{Atom, AtomDataRegistry};
    use dem_atom::{DemAtom, MaterialTable};
    use nalgebra::UnitQuaternion;

    fn push_test_atom(atom: &mut Atom, dem: &mut DemAtom, tag: u32, pos_x: f64, pos_y: f64, pos_z: f64, radius: f64) {
        atom.tag.push(tag);
        atom.atom_type.push(0);
        atom.origin_index.push(0);
        atom.pos_x.push(pos_x); atom.pos_y.push(pos_y); atom.pos_z.push(pos_z);
        atom.vel_x.push(0.0); atom.vel_y.push(0.0); atom.vel_z.push(0.0);
        atom.force_x.push(0.0); atom.force_y.push(0.0); atom.force_z.push(0.0);
        atom.torque_x.push(0.0); atom.torque_y.push(0.0); atom.torque_z.push(0.0);
        atom.mass.push(2500.0 * 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3));
        atom.skin.push(radius);
        atom.is_ghost.push(false);
        atom.has_ghost.push(false);
        atom.is_collision.push(false);
        atom.quaterion.push(UnitQuaternion::identity());
        atom.omega_x.push(0.0); atom.omega_y.push(0.0); atom.omega_z.push(0.0);
        atom.ang_mom_x.push(0.0); atom.ang_mom_y.push(0.0); atom.ang_mom_z.push(0.0);
        dem.radius.push(radius);
        dem.density.push(2500.0);
    }

    fn make_material_table() -> MaterialTable {
        let mut mt = MaterialTable::new();
        mt.add_material("glass", 8.7e9, 0.3, 0.95, 0.4);
        mt.build_pair_tables();
        mt
    }

    fn make_wall_plane(point_x: f64, point_y: f64, point_z: f64, normal_x: f64, normal_y: f64, normal_z: f64) -> WallPlane {
        let mag = (normal_x * normal_x + normal_y * normal_y + normal_z * normal_z).sqrt();
        WallPlane {
            point_x, point_y, point_z,
            normal_x: normal_x / mag,
            normal_y: normal_y / mag,
            normal_z: normal_z / mag,
            material_index: 0,
            name: None,
            bound_x_low: f64::NEG_INFINITY,
            bound_x_high: f64::INFINITY,
            bound_y_low: f64::NEG_INFINITY,
            bound_y_high: f64::INFINITY,
            bound_z_low: f64::NEG_INFINITY,
            bound_z_high: f64::INFINITY,
        }
    }

    #[test]
    fn wall_repulsive_for_overlap() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Atom at z = 0.0005, wall at z = 0 with normal +z → overlap = 0.001 - 0.0005 = 0.0005
        push_test_atom(&mut atom, &mut dem, 0, 0.01, 0.01, 0.0005, radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = Walls {
            planes: vec![make_wall_plane(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
            active: vec![true],
        };

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Force should push atom in +z direction (away from wall)
        assert!(atom.force_z[0] > 0.0, "atom should be pushed away from wall, got {}", atom.force_z[0]);
        assert!((atom.force_x[0]).abs() < 1e-15);
        assert!((atom.force_y[0]).abs() < 1e-15);
    }

    #[test]
    fn wall_zero_for_no_overlap() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Atom at z = 0.002, wall at z = 0 → distance = 0.002 > radius = 0.001
        push_test_atom(&mut atom, &mut dem, 0, 0.01, 0.01, 0.002, radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = Walls {
            planes: vec![make_wall_plane(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
            active: vec![true],
        };

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!((atom.force_z[0]).abs() < 1e-15);
    }

    #[test]
    fn inactive_wall_applies_no_force() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        push_test_atom(&mut atom, &mut dem, 0, 0.01, 0.01, 0.0005, radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = Walls {
            planes: vec![WallPlane {
                point_x: 0.0, point_y: 0.0, point_z: 0.0,
                normal_x: 0.0, normal_y: 0.0, normal_z: 1.0,
                material_index: 0,
                name: Some("blocker".into()),
                bound_x_low: f64::NEG_INFINITY, bound_x_high: f64::INFINITY,
                bound_y_low: f64::NEG_INFINITY, bound_y_high: f64::INFINITY,
                bound_z_low: f64::NEG_INFINITY, bound_z_high: f64::INFINITY,
            }],
            active: vec![false],
        };

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!((atom.force_z[0]).abs() < 1e-15, "inactive wall should apply no force");
    }

    #[test]
    fn angled_wall_force_direction() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // 45-degree wall in x-z plane: normal = (1, 0, 1) normalized
        // Wall passes through origin. Place atom at (0.0003, 0, 0.0003) — distance along normal ≈ 0.000424
        push_test_atom(&mut atom, &mut dem, 0, 0.0003, 0.0, 0.0003, radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let walls = Walls {
            planes: vec![make_wall_plane(0.0, 0.0, 0.0, 1.0, 0.0, 1.0)],
            active: vec![true],
        };

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Force should be along the (1,0,1) normal direction — equal x and z components
        assert!(atom.force_x[0] > 0.0, "force_x should be positive, got {}", atom.force_x[0]);
        assert!(atom.force_z[0] > 0.0, "force_z should be positive, got {}", atom.force_z[0]);
        assert!((atom.force_x[0] - atom.force_z[0]).abs() < 1e-10,
            "force_x and force_z should be equal for 45-degree wall");
        assert!((atom.force_y[0]).abs() < 1e-15);
    }

    #[test]
    fn bounded_wall_ignores_out_of_bounds_atom() {
        let mut atom = Atom::new();
        let mut dem = DemAtom::new();
        let radius = 0.001;

        // Atom at z = 0.0005 overlaps wall at z=0, but atom is at x=0.05 outside bound_x_high=0.04
        push_test_atom(&mut atom, &mut dem, 0, 0.05, 0.01, 0.0005, radius);
        atom.nlocal = 1;
        atom.natoms = 1;

        let mut registry = AtomDataRegistry::new();
        registry.register(dem);

        let mut wall = make_wall_plane(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        wall.bound_x_low = 0.0;
        wall.bound_x_high = 0.04;

        let walls = Walls {
            planes: vec![wall],
            active: vec![true],
        };

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(registry);
        app.add_resource(make_material_table());
        app.add_resource(walls);
        app.add_update_system(wall_contact_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!((atom.force_z[0]).abs() < 1e-15, "out-of-bounds atom should get no wall force");
    }
}

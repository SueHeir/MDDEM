use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use crate::{Atom, AtomDataRegistry, CommBackend, CommResource, Config};

fn default_one_f64() -> f64 {
    1.0
}
fn default_true() -> bool {
    true
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DomainConfig {
    #[serde(default)]
    pub x_low: f64,
    #[serde(default = "default_one_f64")]
    pub x_high: f64,
    #[serde(default)]
    pub y_low: f64,
    #[serde(default = "default_one_f64")]
    pub y_high: f64,
    #[serde(default)]
    pub z_low: f64,
    #[serde(default = "default_one_f64")]
    pub z_high: f64,
    #[serde(default = "default_true")]
    pub periodic_x: bool,
    #[serde(default = "default_true")]
    pub periodic_y: bool,
    #[serde(default = "default_true")]
    pub periodic_z: bool,
}

impl Default for DomainConfig {
    fn default() -> Self {
        DomainConfig {
            x_low: 0.0,
            x_high: 1.0,
            y_low: 0.0,
            y_high: 1.0,
            z_low: 0.0,
            z_high: 1.0,
            periodic_x: true,
            periodic_y: true,
            periodic_z: true,
        }
    }
}

pub struct Domain {
    pub boundaries_low: Vector3<f64>,
    pub boundaries_high: Vector3<f64>,
    pub sub_domain_low: Vector3<f64>,
    pub sub_domain_high: Vector3<f64>,
    pub sub_length: Vector3<f64>,
    pub volume: f64,
    pub size: Vector3<f64>,
    pub is_periodic: Vector3<bool>,
    /// Ghost atom communication cutoff. 0 = use per-atom skin * 4.0 (DEM default).
    pub ghost_cutoff: f64,
}

impl Default for Domain {
    fn default() -> Self {
        Self::new()
    }
}

impl Domain {
    pub fn new() -> Self {
        Domain {
            boundaries_high: Vector3::new(1.0, 1.0, 1.0),
            boundaries_low: Vector3::new(0.0, 0.0, 0.0),
            sub_domain_low: Vector3::new(0.0, 0.0, 0.0),
            sub_domain_high: Vector3::new(1.0, 1.0, 1.0),
            sub_length: Vector3::new(1.0, 1.0, 1.0),
            size: Vector3::new(1.0, 1.0, 1.0),
            is_periodic: Vector3::new(false, false, false),
            volume: 1.0,
            ghost_cutoff: 0.0,
        }
    }
}

// ── DomainDecomposition trait ────────────────────────────────────────────────

pub trait DomainDecomposition: Send + Sync + 'static {
    fn decompose(&self, config: &DomainConfig, comm: &dyn CommBackend) -> Domain;
}

pub struct CartesianDecomposition;

impl DomainDecomposition for CartesianDecomposition {
    fn decompose(&self, config: &DomainConfig, comm: &dyn CommBackend) -> Domain {
        let boundaries_low = Vector3::new(config.x_low, config.y_low, config.z_low);
        let boundaries_high = Vector3::new(config.x_high, config.y_high, config.z_high);
        let size = boundaries_high - boundaries_low;
        let is_periodic = Vector3::new(config.periodic_x, config.periodic_y, config.periodic_z);

        let proc_decomp = comm.processor_decomposition();
        let proc_pos = comm.processor_position();

        let delta_x = size.x / proc_decomp[0] as f64;
        let delta_y = size.y / proc_decomp[1] as f64;
        let delta_z = size.z / proc_decomp[2] as f64;

        let sub_domain_low = Vector3::new(
            boundaries_low.x + delta_x * proc_pos.x as f64,
            boundaries_low.y + delta_y * proc_pos.y as f64,
            boundaries_low.z + delta_z * proc_pos.z as f64,
        );
        let sub_domain_high = Vector3::new(
            boundaries_low.x + delta_x * (1 + proc_pos.x) as f64,
            boundaries_low.y + delta_y * (1 + proc_pos.y) as f64,
            boundaries_low.z + delta_z * (1 + proc_pos.z) as f64,
        );
        let sub_length = Vector3::new(delta_x, delta_y, delta_z);

        Domain {
            boundaries_low,
            boundaries_high,
            sub_domain_low,
            sub_domain_high,
            sub_length,
            size,
            is_periodic,
            volume: size.x * size.y * size.z,
            ghost_cutoff: 0.0,
        }
    }
}

// ── Plugin ───────────────────────────────────────────────────────────────────

pub struct DecompositionResource(pub Box<dyn DomainDecomposition>);

impl std::ops::Deref for DecompositionResource {
    type Target = dyn DomainDecomposition;
    fn deref(&self) -> &(dyn DomainDecomposition + 'static) {
        &*self.0
    }
}

pub struct DomainPlugin {
    decomposition: std::sync::Mutex<Option<Box<dyn DomainDecomposition>>>,
}

impl DomainPlugin {
    pub fn new(decomposition: Box<dyn DomainDecomposition>) -> Self {
        DomainPlugin {
            decomposition: std::sync::Mutex::new(Some(decomposition)),
        }
    }
}

impl Default for DomainPlugin {
    fn default() -> Self {
        DomainPlugin::new(Box::new(CartesianDecomposition))
    }
}

impl Plugin for DomainPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[domain]
# Simulation box boundaries
x_low = 0.0
x_high = 1.0
y_low = 0.0
y_high = 1.0
z_low = 0.0
z_high = 1.0
# Periodic boundary conditions per axis
periodic_x = true
periodic_y = true
periodic_z = true"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<DomainConfig>(app, "domain");

        let decomp = self
            .decomposition
            .lock()
            .unwrap()
            .take()
            .expect("DomainPlugin::build called twice");
        app.add_resource(DecompositionResource(decomp))
            .add_resource(Domain::new())
            .add_setup_system(domain_read_input, ScheduleSetupSet::Setup)
            .add_update_system(pbc, ScheduleSet::PreExchange);
    }
}

pub fn domain_read_input(
    config: Res<DomainConfig>,
    comm: Res<CommResource>,
    decomp: Res<DecompositionResource>,
    mut domain: ResMut<Domain>,
) {
    if comm.rank() == 0 {
        println!(
            "Domain: {} {} {} {} {} {}",
            config.x_low, config.x_high, config.y_low, config.y_high, config.z_low, config.z_high
        );
        println!(
            "Domain: periodic {} {} {}",
            if config.periodic_x { "p" } else { "n" },
            if config.periodic_y { "p" } else { "n" },
            if config.periodic_z { "p" } else { "n" }
        );
    }

    *domain = decomp.decompose(&config, &**comm);
}

pub fn pbc(mut atoms: ResMut<Atom>, domain: Res<Domain>, registry: Res<AtomDataRegistry>) {
    let all_periodic = domain.is_periodic.x && domain.is_periodic.y && domain.is_periodic.z;

    if all_periodic {
        // Fast path: fully periodic, no removals possible — forward iteration
        let low_x = domain.boundaries_low.x;
        let low_y = domain.boundaries_low.y;
        let low_z = domain.boundaries_low.z;
        let sx = domain.size.x;
        let sy = domain.size.y;
        let sz = domain.size.z;
        for i in 0..atoms.len() {
            atoms.pos_x[i] = ((atoms.pos_x[i] - low_x) % sx + sx) % sx + low_x;
            atoms.pos_y[i] = ((atoms.pos_y[i] - low_y) % sy + sy) % sy + low_y;
            atoms.pos_z[i] = ((atoms.pos_z[i] - low_z) % sz + sz) % sz + low_z;
        }
    } else {
        // Slow path: non-periodic axes may require removal
        for i in (0..atoms.len()).rev() {
            if domain.is_periodic.x {
                let low = domain.boundaries_low.x;
                let s = domain.size.x;
                atoms.pos_x[i] = ((atoms.pos_x[i] - low) % s + s) % s + low;
            } else if atoms.pos_x[i] < domain.boundaries_low.x
                || atoms.pos_x[i] >= domain.boundaries_high.x
            {
                atoms.swap_remove(i);
                registry.swap_remove_all(i);
                continue;
            }
            if domain.is_periodic.y {
                let low = domain.boundaries_low.y;
                let s = domain.size.y;
                atoms.pos_y[i] = ((atoms.pos_y[i] - low) % s + s) % s + low;
            } else if atoms.pos_y[i] < domain.boundaries_low.y
                || atoms.pos_y[i] >= domain.boundaries_high.y
            {
                atoms.swap_remove(i);
                registry.swap_remove_all(i);
                continue;
            }
            if domain.is_periodic.z {
                let low = domain.boundaries_low.z;
                let s = domain.size.z;
                atoms.pos_z[i] = ((atoms.pos_z[i] - low) % s + s) % s + low;
            } else if atoms.pos_z[i] < domain.boundaries_low.z
                || atoms.pos_z[i] >= domain.boundaries_high.z
            {
                atoms.swap_remove(i);
                registry.swap_remove_all(i);
                continue;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SingleProcessComm;

    fn make_comm(decomp: Vector3<i32>, pos: Vector3<i32>) -> SingleProcessComm {
        let mut c = SingleProcessComm::new();
        c.set_processor_grid(decomp, pos);
        c
    }

    #[test]
    fn cartesian_single_proc_full_domain() {
        let config = DomainConfig {
            x_low: 0.0,
            x_high: 10.0,
            y_low: 0.0,
            y_high: 5.0,
            z_low: 0.0,
            z_high: 2.0,
            periodic_x: true,
            periodic_y: false,
            periodic_z: true,
        };
        let comm = make_comm(Vector3::new(1, 1, 1), Vector3::new(0, 0, 0));
        let domain = CartesianDecomposition.decompose(&config, &comm);

        assert_eq!(domain.boundaries_low, Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(domain.boundaries_high, Vector3::new(10.0, 5.0, 2.0));
        assert_eq!(domain.sub_domain_low, Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(domain.sub_domain_high, Vector3::new(10.0, 5.0, 2.0));
        assert_eq!(domain.is_periodic, Vector3::new(true, false, true));
        assert!((domain.volume - 100.0).abs() < 1e-10);
    }

    #[test]
    fn cartesian_multi_proc_subdivides() {
        let config = DomainConfig {
            x_low: 0.0,
            x_high: 10.0,
            y_low: 0.0,
            y_high: 10.0,
            z_low: 0.0,
            z_high: 10.0,
            periodic_x: true,
            periodic_y: true,
            periodic_z: true,
        };
        // Simulate proc at position (1,0,0) in a 2x1x1 decomposition
        let comm = make_comm(Vector3::new(2, 1, 1), Vector3::new(1, 0, 0));
        let domain = CartesianDecomposition.decompose(&config, &comm);

        assert!((domain.sub_domain_low.x - 5.0).abs() < 1e-10);
        assert!((domain.sub_domain_high.x - 10.0).abs() < 1e-10);
        assert!((domain.sub_length.x - 5.0).abs() < 1e-10);
    }
}

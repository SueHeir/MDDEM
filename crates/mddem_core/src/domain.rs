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
#[serde(deny_unknown_fields)]
/// TOML `[domain]` — simulation box boundaries and periodic flags.
pub struct DomainConfig {
    /// Lower x boundary of the simulation box.
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

/// Simulation box geometry: global boundaries, sub-domain bounds, and periodicity.
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
    /// When true, PBC boundary crossings force a full ghost + neighbor rebuild.
    /// Required for DEM (contact history depends on correct ghost identity).
    /// Safe to leave false for pair potentials like LJ where stale ghosts are harmless.
    pub pbc_strict: bool,
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
            pbc_strict: false,
        }
    }
}

// ── DomainDecomposition trait ────────────────────────────────────────────────

/// Computes sub-domain bounds from config and processor grid.
pub trait DomainDecomposition: Send + Sync + 'static {
    fn decompose(&self, config: &DomainConfig, comm: &dyn CommBackend) -> Domain;
}

/// Uniform Cartesian grid decomposition (default).
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
            pbc_strict: false,
        }
    }
}

// ── Plugin ───────────────────────────────────────────────────────────────────

/// Wraps a [`DomainDecomposition`] implementation, used as `Res<DecompositionResource>`.
pub struct DecompositionResource(pub Box<dyn DomainDecomposition>);

impl std::ops::Deref for DecompositionResource {
    type Target = dyn DomainDecomposition;
    fn deref(&self) -> &(dyn DomainDecomposition + 'static) {
        &*self.0
    }
}

/// Registers [`Domain`] resource and periodic boundary condition system.
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

/// Wrap a position into [low, low+size) with periodic boundaries.
#[inline]
fn wrap_periodic(pos: f64, low: f64, size: f64) -> f64 {
    ((pos - low) % size + size) % size + low
}

pub fn pbc(mut atoms: ResMut<Atom>, domain: Res<Domain>, registry: Res<AtomDataRegistry>) {
    let low = domain.boundaries_low;
    let high = domain.boundaries_high;
    let size = domain.size;
    let periodic = domain.is_periodic;

    if periodic.x && periodic.y && periodic.z {
        // Fast path: fully periodic, no removals possible (local atoms only, ghosts live outside box)
        for i in 0..atoms.nlocal as usize {
            atoms.pos[i][0] = wrap_periodic(atoms.pos[i][0], low.x, size.x);
            atoms.pos[i][1] = wrap_periodic(atoms.pos[i][1], low.y, size.y);
            atoms.pos[i][2] = wrap_periodic(atoms.pos[i][2], low.z, size.z);
        }
    } else {
        // Slow path: non-periodic axes may require removal (local atoms only)
        'outer: for i in (0..atoms.nlocal as usize).rev() {
            macro_rules! handle_dim {
                ($pos:expr, $is_periodic:expr, $lo:expr, $hi:expr, $sz:expr) => {
                    if $is_periodic {
                        $pos = wrap_periodic($pos, $lo, $sz);
                    } else if $pos < $lo || $pos >= $hi {
                        atoms.swap_remove(i);
                        registry.swap_remove_all(i);
                        continue 'outer;
                    }
                };
            }
            handle_dim!(atoms.pos[i][0], periodic.x, low.x, high.x, size.x);
            handle_dim!(atoms.pos[i][1], periodic.y, low.y, high.y, size.y);
            handle_dim!(atoms.pos[i][2], periodic.z, low.z, high.z, size.z);
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

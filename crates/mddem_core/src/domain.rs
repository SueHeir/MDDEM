//! Simulation box geometry, domain decomposition, and periodic boundary conditions.
//!
//! This module provides:
//! - [`Domain`]: runtime state for box boundaries, sub-domain bounds, and periodicity
//! - [`DomainConfig`]: TOML `[domain]` section with boundary types and box extents
//! - [`DomainDecomposition`] trait and [`CartesianDecomposition`]: split the box
//!   across MPI ranks
//! - [`DomainPlugin`]: registers setup, PBC wrapping, and shrink-wrap systems

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{Atom, AtomDataRegistry, CommBackend, CommResource, Config};

fn default_one_f64() -> f64 {
    1.0
}
fn default_true() -> bool {
    true
}

/// Boundary condition type for a single axis.
///
/// - `Periodic`: atoms that exit one side re-enter from the opposite side.
/// - `Fixed`: the box boundary is static; atoms leaving are removed.
/// - `ShrinkWrap`: the box boundary automatically adjusts each step to
///   encompass all atoms (plus padding), like LAMMPS `boundary s`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BoundaryType {
    Periodic,
    Fixed,
    ShrinkWrap,
}

impl Default for BoundaryType {
    fn default() -> Self {
        BoundaryType::Periodic
    }
}

fn default_boundary() -> Option<BoundaryType> {
    None
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[domain]` — simulation box boundaries and periodic flags.
pub struct DomainConfig {
    /// Lower x boundary of the simulation box (simulation units, e.g. meters for DEM).
    #[serde(default, alias = "x_lo")]
    pub x_low: f64,
    /// Upper x boundary of the simulation box (simulation units).
    #[serde(default = "default_one_f64", alias = "x_hi")]
    pub x_high: f64,
    /// Lower y boundary of the simulation box (simulation units).
    #[serde(default, alias = "y_lo")]
    pub y_low: f64,
    /// Upper y boundary of the simulation box (simulation units).
    #[serde(default = "default_one_f64", alias = "y_hi")]
    pub y_high: f64,
    /// Lower z boundary of the simulation box (simulation units).
    #[serde(default, alias = "z_lo")]
    pub z_low: f64,
    /// Upper z boundary of the simulation box (simulation units).
    #[serde(default = "default_one_f64", alias = "z_hi")]
    pub z_high: f64,
    /// Whether the x axis uses periodic boundary conditions.
    /// Ignored when `boundary_x` is set.
    #[serde(default = "default_true", alias = "x_periodic")]
    pub periodic_x: bool,
    /// Whether the y axis uses periodic boundary conditions.
    /// Ignored when `boundary_y` is set.
    #[serde(default = "default_true", alias = "y_periodic")]
    pub periodic_y: bool,
    /// Whether the z axis uses periodic boundary conditions.
    /// Ignored when `boundary_z` is set.
    #[serde(default = "default_true", alias = "z_periodic")]
    pub periodic_z: bool,
    /// Explicit boundary type for the x axis: "periodic", "fixed", or "shrink-wrap".
    /// When set, overrides `periodic_x`.
    #[serde(default = "default_boundary")]
    pub boundary_x: Option<BoundaryType>,
    /// Explicit boundary type for the y axis: "periodic", "fixed", or "shrink-wrap".
    /// When set, overrides `periodic_y`.
    #[serde(default = "default_boundary")]
    pub boundary_y: Option<BoundaryType>,
    /// Explicit boundary type for the z axis: "periodic", "fixed", or "shrink-wrap".
    /// When set, overrides `periodic_z`.
    #[serde(default = "default_boundary")]
    pub boundary_z: Option<BoundaryType>,
    /// Padding added on each side of shrink-wrap boundaries (simulation units).
    /// Defaults to 0.0 (auto-computed from max particle cutoff radius).
    #[serde(default)]
    pub shrink_wrap_padding: f64,
}

impl DomainConfig {
    /// Resolve the effective boundary type per axis, considering both
    /// the legacy `periodic_*` booleans and the new `boundary_*` fields.
    pub fn resolved_boundary_types(&self) -> [BoundaryType; 3] {
        let resolve = |boundary: Option<BoundaryType>, periodic: bool| -> BoundaryType {
            if let Some(bt) = boundary {
                bt
            } else if periodic {
                BoundaryType::Periodic
            } else {
                BoundaryType::Fixed
            }
        };
        [
            resolve(self.boundary_x, self.periodic_x),
            resolve(self.boundary_y, self.periodic_y),
            resolve(self.boundary_z, self.periodic_z),
        ]
    }
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
            boundary_x: None,
            boundary_y: None,
            boundary_z: None,
            shrink_wrap_padding: 0.0,
        }
    }
}

/// Simulation box geometry: global boundaries, sub-domain bounds, and periodicity.
pub struct Domain {
    pub boundaries_low: [f64; 3],
    pub boundaries_high: [f64; 3],
    pub sub_domain_low: [f64; 3],
    pub sub_domain_high: [f64; 3],
    pub sub_length: [f64; 3],
    pub volume: f64,
    pub size: [f64; 3],
    pub is_periodic: [bool; 3],
    /// Per-axis boundary type.
    pub boundary_type: [BoundaryType; 3],
    /// Per-axis shrink-wrap flag (convenience: `boundary_type[d] == ShrinkWrap`).
    pub is_shrink_wrap: [bool; 3],
    /// Padding for shrink-wrap boundaries. If 0, uses ghost_cutoff as padding.
    pub shrink_wrap_padding: f64,
    /// Set to true whenever shrink-wrap updates domain bounds.
    /// Cleared by the neighbor system after it recomputes bins.
    pub bounds_changed: bool,
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
            boundaries_high: [1.0; 3],
            boundaries_low: [0.0; 3],
            sub_domain_low: [0.0; 3],
            sub_domain_high: [1.0; 3],
            sub_length: [1.0; 3],
            size: [1.0; 3],
            is_periodic: [false; 3],
            boundary_type: [BoundaryType::Periodic; 3],
            is_shrink_wrap: [false; 3],
            shrink_wrap_padding: 0.0,
            bounds_changed: false,
            volume: 1.0,
            ghost_cutoff: 0.0,
            pbc_strict: false,
        }
    }

    /// Recompute derived fields (size, sub_length, volume) after bounds change.
    pub fn update_derived(&mut self) {
        for d in 0..3 {
            self.size[d] = self.boundaries_high[d] - self.boundaries_low[d];
            self.sub_length[d] = self.sub_domain_high[d] - self.sub_domain_low[d];
        }
        self.volume = self.size[0] * self.size[1] * self.size[2];
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
        let boundaries_low = [config.x_low, config.y_low, config.z_low];
        let boundaries_high = [config.x_high, config.y_high, config.z_high];
        let size = [
            boundaries_high[0] - boundaries_low[0],
            boundaries_high[1] - boundaries_low[1],
            boundaries_high[2] - boundaries_low[2],
        ];
        let boundary_types = config.resolved_boundary_types();
        let is_periodic = [
            boundary_types[0] == BoundaryType::Periodic,
            boundary_types[1] == BoundaryType::Periodic,
            boundary_types[2] == BoundaryType::Periodic,
        ];
        let is_shrink_wrap = [
            boundary_types[0] == BoundaryType::ShrinkWrap,
            boundary_types[1] == BoundaryType::ShrinkWrap,
            boundary_types[2] == BoundaryType::ShrinkWrap,
        ];

        let proc_decomp = comm.processor_decomposition();
        let proc_pos = comm.processor_position();

        let delta_x = size[0] / proc_decomp[0] as f64;
        let delta_y = size[1] / proc_decomp[1] as f64;
        let delta_z = size[2] / proc_decomp[2] as f64;

        let sub_domain_low = [
            boundaries_low[0] + delta_x * proc_pos[0] as f64,
            boundaries_low[1] + delta_y * proc_pos[1] as f64,
            boundaries_low[2] + delta_z * proc_pos[2] as f64,
        ];
        let sub_domain_high = [
            boundaries_low[0] + delta_x * (1 + proc_pos[0]) as f64,
            boundaries_low[1] + delta_y * (1 + proc_pos[1]) as f64,
            boundaries_low[2] + delta_z * (1 + proc_pos[2]) as f64,
        ];
        let sub_length = [delta_x, delta_y, delta_z];

        Domain {
            boundaries_low,
            boundaries_high,
            sub_domain_low,
            sub_domain_high,
            sub_length,
            size,
            is_periodic,
            boundary_type: boundary_types,
            is_shrink_wrap,
            shrink_wrap_padding: config.shrink_wrap_padding,
            bounds_changed: false,
            volume: size[0] * size[1] * size[2],
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
# Simulation box boundaries (also accepts x_lo/x_hi, y_lo/y_hi, z_lo/z_hi)
x_low = 0.0
x_high = 1.0
y_low = 0.0
y_high = 1.0
z_low = 0.0
z_high = 1.0
# Periodic boundary conditions per axis (also accepts x_periodic, y_periodic, z_periodic)
periodic_x = true
periodic_y = true
periodic_z = true
# Explicit boundary type per axis: "periodic", "fixed", or "shrink-wrap"
# When set, overrides the corresponding periodic_* flag.
# boundary_x = "shrink-wrap"
# boundary_y = "shrink-wrap"
# boundary_z = "shrink-wrap"
# Padding for shrink-wrap boundaries [simulation units]. 0 = auto (use ghost cutoff).
# shrink_wrap_padding = 0.0"#,
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
            .add_update_system(
                shrink_wrap.label("shrink_wrap").before("pbc"),
                ScheduleSet::PreExchange,
            )
            .add_update_system(pbc.label("pbc"), ScheduleSet::PreExchange);
    }
}

fn boundary_type_char(bt: BoundaryType) -> char {
    match bt {
        BoundaryType::Periodic => 'p',
        BoundaryType::Fixed => 'f',
        BoundaryType::ShrinkWrap => 's',
    }
}

/// Setup system: read `[domain]` config and initialize the [`Domain`] resource
/// via the registered [`DomainDecomposition`].
pub fn domain_read_input(
    config: Res<DomainConfig>,
    comm: Res<CommResource>,
    decomp: Res<DecompositionResource>,
    mut domain: ResMut<Domain>,
) {
    let boundary_types = config.resolved_boundary_types();

    let has_shrink_wrap = boundary_types.contains(&BoundaryType::ShrinkWrap);

    // Shrink-wrap is not yet supported with MPI — fail early to prevent silent wrong results.
    // MPI support requires correct sub-domain bound updates and global reductions across ranks.
    if comm.size() > 1 && has_shrink_wrap {
        panic!("Shrink-wrap boundaries are not yet supported with MPI (nprocs > 1). \
                Use fixed or periodic boundaries, or run with a single process.");
    }

    if comm.rank() == 0 {
        println!(
            "Domain: {} {} {} {} {} {}",
            config.x_low, config.x_high, config.y_low, config.y_high, config.z_low, config.z_high
        );
        println!(
            "Domain: boundary {} {} {}",
            boundary_type_char(boundary_types[0]),
            boundary_type_char(boundary_types[1]),
            boundary_type_char(boundary_types[2]),
        );
        if has_shrink_wrap {
            if config.shrink_wrap_padding > 0.0 {
                println!("Domain: shrink-wrap padding = {}", config.shrink_wrap_padding);
            } else {
                println!("Domain: shrink-wrap padding = auto (ghost cutoff)");
            }
        }
    }

    *domain = decomp.decompose(&config, &**comm);
}

/// Core shrink-wrap logic: update domain bounds to encompass all atom positions + padding.
///
/// Returns `true` if any bounds changed. This is extracted as a standalone function
/// so it can be unit-tested without the ECS resource wrappers.
///
/// **MPI note**: Shrink-wrap is currently single-process only (see the panic in
/// `domain_read_input`). Future MPI support would need `all_reduce_min/max` for
/// global extremes and correct sub-domain bound updates per rank.
pub fn shrink_wrap_update(domain: &mut Domain, positions: &[[f64; 3]], nlocal: usize) -> bool {
    let any_shrink = domain.is_shrink_wrap[0] || domain.is_shrink_wrap[1] || domain.is_shrink_wrap[2];
    if !any_shrink || nlocal == 0 {
        return false;
    }

    // Padding: use explicit value if > 0, otherwise fall back to ghost_cutoff.
    // ghost_cutoff encompasses the max pairwise interaction distance + skin buffer,
    // so it is a conservative but safe choice for any unit system.
    let padding = if domain.shrink_wrap_padding > 0.0 {
        domain.shrink_wrap_padding
    } else if domain.ghost_cutoff > 0.0 {
        domain.ghost_cutoff
    } else {
        // ghost_cutoff not yet set (shrink_wrap runs before neighbor_setup on first step).
        // Use 1% of the current domain size as a unit-independent fallback.
        let max_size = domain.size[0].max(domain.size[1]).max(domain.size[2]);
        if max_size > 0.0 { max_size * 0.01 } else { 1.0 }
    };

    let mut changed = false;

    for d in 0..3 {
        if !domain.is_shrink_wrap[d] {
            continue;
        }

        // Find min/max positions on this axis
        let mut pos_min = f64::MAX;
        let mut pos_max = f64::MIN;
        for pos in &positions[..nlocal] {
            let p = pos[d];
            if p < pos_min {
                pos_min = p;
            }
            if p > pos_max {
                pos_max = p;
            }
        }

        let new_low = pos_min - padding;
        let new_high = pos_max + padding;

        // Check if bounds actually changed (with tolerance to avoid churn)
        let tol = 1e-12;
        if (new_low - domain.boundaries_low[d]).abs() > tol
            || (new_high - domain.boundaries_high[d]).abs() > tol
        {
            domain.boundaries_low[d] = new_low;
            domain.boundaries_high[d] = new_high;
            // Single-process: sub-domain = global domain.
            // TODO: For MPI, sub-domain bounds must be recomputed via domain decomposition.
            domain.sub_domain_low[d] = new_low;
            domain.sub_domain_high[d] = new_high;
            changed = true;
        }
    }

    if changed {
        domain.update_derived();
        domain.bounds_changed = true;
    }
    changed
}

/// ECS system wrapper: update shrink-wrap boundaries each step.
///
/// Runs at `PreExchange` before `pbc`. Delegates to [`shrink_wrap_update`].
pub fn shrink_wrap(
    atoms: Res<Atom>,
    mut domain: ResMut<Domain>,
) {
    shrink_wrap_update(&mut domain, &atoms.pos, atoms.nlocal as usize);
}

/// Wrap a position into [low, low+size) with periodic boundaries.
#[inline]
fn wrap_periodic(mut pos: f64, low: f64, size: f64) -> f64 {
    let high = low + size;
    if pos < low {
        pos += size;
    } else if pos >= high {
        pos -= size;
    }
    pos
}

/// Apply periodic boundary conditions: wrap positions on periodic axes,
/// remove out-of-bounds atoms on fixed/shrink-wrap axes.
pub fn pbc(mut atoms: ResMut<Atom>, domain: Res<Domain>, registry: Res<AtomDataRegistry>) {
    let low = domain.boundaries_low;
    let high = domain.boundaries_high;
    let size = domain.size;
    let periodic = domain.is_periodic;

    if periodic[0] && periodic[1] && periodic[2] {
        // Fast path: fully periodic, no removals possible (local atoms only, ghosts live outside box)
        for i in 0..atoms.nlocal as usize {
            atoms.pos[i][0] = wrap_periodic(atoms.pos[i][0], low[0], size[0]);
            atoms.pos[i][1] = wrap_periodic(atoms.pos[i][1], low[1], size[1]);
            atoms.pos[i][2] = wrap_periodic(atoms.pos[i][2], low[2], size[2]);
        }
    } else {
        // Slow path: non-periodic axes may require removal (local atoms only).
        // Shrink-wrap axes: atoms are always inside bounds (shrink_wrap ran first),
        // so the out-of-bounds check is a no-op — but it's harmless and correct.
        let nlocal_before = atoms.nlocal as usize;
        let mut removed = 0usize;
        'outer: for i in (0..atoms.nlocal as usize).rev() {
            macro_rules! handle_dim {
                ($pos:expr, $is_periodic:expr, $lo:expr, $hi:expr, $sz:expr) => {
                    if $is_periodic {
                        $pos = wrap_periodic($pos, $lo, $sz);
                    } else if $pos < $lo || $pos >= $hi {
                        atoms.swap_remove(i);
                        registry.swap_remove_all(i);
                        removed += 1;
                        continue 'outer;
                    }
                };
            }
            handle_dim!(atoms.pos[i][0], periodic[0], low[0], high[0], size[0]);
            handle_dim!(atoms.pos[i][1], periodic[1], low[1], high[1], size[1]);
            handle_dim!(atoms.pos[i][2], periodic[2], low[2], high[2], size[2]);
        }
        // Update nlocal and invalidate ghost communication sendlists so that
        // `borders` performs a full rebuild instead of using stale indices.
        if removed > 0 {
            atoms.nlocal = (nlocal_before - removed) as u32;
            atoms.communicate_only = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SingleProcessComm;

    fn make_comm(decomp: [i32; 3], pos: [i32; 3]) -> SingleProcessComm {
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
            ..Default::default()
        };
        let comm = make_comm([1, 1, 1], [0, 0, 0]);
        let domain = CartesianDecomposition.decompose(&config, &comm);

        assert_eq!(domain.boundaries_low, [0.0, 0.0, 0.0]);
        assert_eq!(domain.boundaries_high, [10.0, 5.0, 2.0]);
        assert_eq!(domain.sub_domain_low, [0.0, 0.0, 0.0]);
        assert_eq!(domain.sub_domain_high, [10.0, 5.0, 2.0]);
        assert_eq!(domain.is_periodic, [true, false, true]);
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
            ..Default::default()
        };
        // Simulate proc at position (1,0,0) in a 2x1x1 decomposition
        let comm = make_comm([2, 1, 1], [1, 0, 0]);
        let domain = CartesianDecomposition.decompose(&config, &comm);

        assert!((domain.sub_domain_low[0] - 5.0).abs() < 1e-10);
        assert!((domain.sub_domain_high[0] - 10.0).abs() < 1e-10);
        assert!((domain.sub_length[0] - 5.0).abs() < 1e-10);
    }

    // ── Boundary type resolution tests ──────────────────────────────────────

    #[test]
    fn backward_compat_periodic_flags() {
        // Old-style config: only periodic_* flags, no boundary_* fields
        let config = DomainConfig {
            periodic_x: true,
            periodic_y: false,
            periodic_z: true,
            ..Default::default()
        };
        let types = config.resolved_boundary_types();
        assert_eq!(types, [BoundaryType::Periodic, BoundaryType::Fixed, BoundaryType::Periodic]);
    }

    #[test]
    fn boundary_type_overrides_periodic() {
        // boundary_z = shrink-wrap should override periodic_z = true
        let config = DomainConfig {
            periodic_x: true,
            periodic_y: true,
            periodic_z: true, // would be Periodic...
            boundary_z: Some(BoundaryType::ShrinkWrap), // ...but overridden
            ..Default::default()
        };
        let types = config.resolved_boundary_types();
        assert_eq!(types[0], BoundaryType::Periodic);
        assert_eq!(types[1], BoundaryType::Periodic);
        assert_eq!(types[2], BoundaryType::ShrinkWrap);
    }

    #[test]
    fn shrink_wrap_decomposition() {
        let config = DomainConfig {
            x_low: 0.0,
            x_high: 10.0,
            y_low: 0.0,
            y_high: 10.0,
            z_low: 0.0,
            z_high: 20.0,
            periodic_x: true,
            periodic_y: true,
            periodic_z: false,
            boundary_z: Some(BoundaryType::ShrinkWrap),
            ..Default::default()
        };
        let comm = make_comm([1, 1, 1], [0, 0, 0]);
        let domain = CartesianDecomposition.decompose(&config, &comm);

        assert_eq!(domain.is_periodic, [true, true, false]);
        assert_eq!(domain.is_shrink_wrap, [false, false, true]);
        assert_eq!(domain.boundary_type[2], BoundaryType::ShrinkWrap);
    }

    #[test]
    fn toml_parse_boundary_types() {
        let toml_str = r#"
            x_low = 0.0
            x_high = 1.0
            y_low = 0.0
            y_high = 1.0
            z_low = 0.0
            z_high = 1.0
            periodic_x = true
            periodic_y = true
            periodic_z = false
            boundary_z = "shrink-wrap"
        "#;
        let config: DomainConfig = toml::from_str(toml_str).unwrap();
        let types = config.resolved_boundary_types();
        assert_eq!(types[2], BoundaryType::ShrinkWrap);
    }

    #[test]
    fn toml_parse_backward_compat() {
        // Config without boundary_* fields should still work
        let toml_str = r#"
            x_low = 0.0
            x_high = 1.0
            y_low = 0.0
            y_high = 1.0
            z_low = 0.0
            z_high = 1.0
            periodic_x = true
            periodic_y = false
            periodic_z = true
        "#;
        let config: DomainConfig = toml::from_str(toml_str).unwrap();
        let types = config.resolved_boundary_types();
        assert_eq!(types, [BoundaryType::Periodic, BoundaryType::Fixed, BoundaryType::Periodic]);
    }

    // ── Shrink-wrap update tests ────────────────────────────────────────────

    // ── Shrink-wrap system tests ───────────────────────────────────────────

    #[test]
    fn shrink_wrap_expands_to_atom_positions() {
        let mut domain = Domain::new();
        domain.boundaries_low = [0.0, 0.0, 0.0];
        domain.boundaries_high = [10.0, 10.0, 10.0];
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [10.0, 10.0, 10.0];
        domain.size = [10.0, 10.0, 10.0];
        domain.sub_length = [10.0, 10.0, 10.0];
        domain.is_shrink_wrap = [false, false, true]; // only z is shrink-wrap
        domain.shrink_wrap_padding = 0.5;

        let positions = vec![
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 8.0],
            [9.0, 1.0, 1.0],
        ];

        let changed = super::shrink_wrap_update(&mut domain, &positions, 3);
        assert!(changed);
        // z bounds should wrap to [min_z - padding, max_z + padding] = [1.0 - 0.5, 8.0 + 0.5]
        assert!((domain.boundaries_low[2] - 0.5).abs() < 1e-10);
        assert!((domain.boundaries_high[2] - 8.5).abs() < 1e-10);
        // x and y should be unchanged (not shrink-wrap)
        assert!((domain.boundaries_low[0] - 0.0).abs() < 1e-10);
        assert!((domain.boundaries_high[0] - 10.0).abs() < 1e-10);
        assert!(domain.bounds_changed);
    }

    #[test]
    fn shrink_wrap_no_change_when_within_tolerance() {
        // Set bounds to exactly match what shrink_wrap would compute:
        // z positions are [3.0, 8.0, 1.0], so min=1.0, max=8.0
        // With padding=0.5: low=0.5, high=8.5
        let mut domain = Domain::new();
        domain.boundaries_low = [0.0, 0.0, 0.5];
        domain.boundaries_high = [10.0, 10.0, 8.5];
        domain.sub_domain_low = [0.0, 0.0, 0.5];
        domain.sub_domain_high = [10.0, 10.0, 8.5];
        domain.size = [10.0, 10.0, 8.0];
        domain.sub_length = [10.0, 10.0, 8.0];
        domain.is_shrink_wrap = [false, false, true];
        domain.shrink_wrap_padding = 0.5;
        domain.bounds_changed = false;

        let positions = vec![
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 8.0],
            [9.0, 1.0, 1.0],
        ];

        let changed = super::shrink_wrap_update(&mut domain, &positions, 3);
        assert!(!changed);
        assert!(!domain.bounds_changed);
    }

    #[test]
    fn shrink_wrap_no_atoms_is_noop() {
        let mut domain = Domain::new();
        domain.is_shrink_wrap = [true, true, true];
        domain.bounds_changed = false;
        let positions: Vec<[f64; 3]> = vec![];

        let changed = super::shrink_wrap_update(&mut domain, &positions, 0);
        assert!(!changed);
    }

    #[test]
    fn shrink_wrap_all_axes() {
        let mut domain = Domain::new();
        domain.boundaries_low = [0.0, 0.0, 0.0];
        domain.boundaries_high = [100.0, 100.0, 100.0];
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [100.0, 100.0, 100.0];
        domain.size = [100.0, 100.0, 100.0];
        domain.sub_length = [100.0, 100.0, 100.0];
        domain.is_shrink_wrap = [true, true, true];
        domain.shrink_wrap_padding = 1.0;

        let positions = vec![
            [10.0, 20.0, 30.0],
            [50.0, 60.0, 70.0],
        ];

        let changed = super::shrink_wrap_update(&mut domain, &positions, 2);
        assert!(changed);
        // x: [10-1, 50+1] = [9, 51]
        assert!((domain.boundaries_low[0] - 9.0).abs() < 1e-10);
        assert!((domain.boundaries_high[0] - 51.0).abs() < 1e-10);
        // y: [20-1, 60+1] = [19, 61]
        assert!((domain.boundaries_low[1] - 19.0).abs() < 1e-10);
        assert!((domain.boundaries_high[1] - 61.0).abs() < 1e-10);
        // z: [30-1, 70+1] = [29, 71]
        assert!((domain.boundaries_low[2] - 29.0).abs() < 1e-10);
        assert!((domain.boundaries_high[2] - 71.0).abs() < 1e-10);
        // Derived fields should be updated
        assert!((domain.size[0] - 42.0).abs() < 1e-10);
        assert!((domain.volume - 42.0 * 42.0 * 42.0).abs() < 1e-6);
    }

    #[test]
    fn shrink_wrap_uses_ghost_cutoff_fallback() {
        let mut domain = Domain::new();
        domain.boundaries_low = [0.0, 0.0, 0.0];
        domain.boundaries_high = [10.0, 10.0, 10.0];
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [10.0, 10.0, 10.0];
        domain.size = [10.0, 10.0, 10.0];
        domain.sub_length = [10.0, 10.0, 10.0];
        domain.is_shrink_wrap = [false, false, true];
        domain.shrink_wrap_padding = 0.0; // no explicit padding
        domain.ghost_cutoff = 2.0; // should use this as fallback

        let positions = vec![[5.0, 5.0, 5.0]];
        super::shrink_wrap_update(&mut domain, &positions, 1);
        // z bounds: [5.0 - 2.0, 5.0 + 2.0] = [3.0, 7.0]
        assert!((domain.boundaries_low[2] - 3.0).abs() < 1e-10);
        assert!((domain.boundaries_high[2] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn domain_update_derived() {
        let mut domain = Domain::new();
        domain.boundaries_low = [0.0, 0.0, 0.0];
        domain.boundaries_high = [5.0, 10.0, 20.0];
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [5.0, 10.0, 20.0];
        domain.update_derived();

        assert!((domain.size[0] - 5.0).abs() < 1e-10);
        assert!((domain.size[1] - 10.0).abs() < 1e-10);
        assert!((domain.size[2] - 20.0).abs() < 1e-10);
        assert!((domain.volume - 1000.0).abs() < 1e-10);
    }
}

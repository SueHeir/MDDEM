//! Simulation box geometry, domain decomposition, and periodic boundary conditions.
//!
//! This module provides:
//! - [`Domain`]: runtime state for box boundaries, sub-domain bounds, and periodicity
//! - [`DomainConfig`]: TOML `[domain]` section with boundary types and box extents
//! - [`decompose_domain`]: uniform Cartesian grid decomposition across MPI ranks
//! - [`DomainPlugin`]: registers setup, PBC wrapping, and shrink-wrap systems

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{Atom, AtomDataRegistry, CommBackend, CommResource, CommState, Config, ParticleSimScheduleSet, ScheduleSetupSet};
use grass_scheduler::prelude::CurrentState;

fn default_one_f64() -> f64 {
    1.0
}
fn default_periodic() -> BoundaryType {
    BoundaryType::Periodic
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

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[domain]` — simulation box boundaries and boundary types.
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
    /// Boundary type for the x axis: "periodic", "fixed", or "shrink-wrap".
    #[serde(default = "default_periodic")]
    pub boundary_x: BoundaryType,
    /// Boundary type for the y axis: "periodic", "fixed", or "shrink-wrap".
    #[serde(default = "default_periodic")]
    pub boundary_y: BoundaryType,
    /// Boundary type for the z axis: "periodic", "fixed", or "shrink-wrap".
    #[serde(default = "default_periodic")]
    pub boundary_z: BoundaryType,
    /// Padding added on each side of shrink-wrap boundaries (simulation units).
    /// Defaults to 0.0 (auto-computed from max particle cutoff radius).
    #[serde(default)]
    pub shrink_wrap_padding: f64,
}

impl DomainConfig {
    /// Return the boundary types as a 3-element array `[x, y, z]`.
    pub fn boundary_types(&self) -> [BoundaryType; 3] {
        [self.boundary_x, self.boundary_y, self.boundary_z]
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
            boundary_x: BoundaryType::Periodic,
            boundary_y: BoundaryType::Periodic,
            boundary_z: BoundaryType::Periodic,
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
    /// Per-axis boundary type (single source of truth).
    pub boundary_type: [BoundaryType; 3],
    /// Padding for shrink-wrap boundaries. If 0, uses ghost_cutoff as padding.
    pub shrink_wrap_padding: f64,
    /// Set to true whenever shrink-wrap updates domain bounds.
    /// Cleared by the neighbor system after it recomputes bins.
    pub bounds_changed: bool,
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
            boundaries_high: [1.0; 3],
            boundaries_low: [0.0; 3],
            sub_domain_low: [0.0; 3],
            sub_domain_high: [1.0; 3],
            sub_length: [1.0; 3],
            size: [1.0; 3],
            boundary_type: [BoundaryType::Periodic; 3],
            shrink_wrap_padding: 0.0,
            bounds_changed: false,
            volume: 1.0,
            ghost_cutoff: 0.0,
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

    /// Whether axis `d` is periodic.
    #[inline]
    pub fn is_periodic(&self, d: usize) -> bool {
        self.boundary_type[d] == BoundaryType::Periodic
    }

    /// Whether axis `d` is shrink-wrap.
    #[inline]
    pub fn is_shrink_wrap(&self, d: usize) -> bool {
        self.boundary_type[d] == BoundaryType::ShrinkWrap
    }

    /// Return periodic flags as a `[bool; 3]` array (convenience for bulk assignment).
    pub fn periodic_flags(&self) -> [bool; 3] {
        [self.is_periodic(0), self.is_periodic(1), self.is_periodic(2)]
    }
}

// ── Domain decomposition ─────────────────────────────────────────────────────

/// Compute [`Domain`] from config using uniform Cartesian grid decomposition.
pub fn decompose_domain(config: &DomainConfig, comm: &dyn CommBackend) -> Domain {
    let boundaries_low = [config.x_low, config.y_low, config.z_low];
    let boundaries_high = [config.x_high, config.y_high, config.z_high];
    let size = [
        boundaries_high[0] - boundaries_low[0],
        boundaries_high[1] - boundaries_low[1],
        boundaries_high[2] - boundaries_low[2],
    ];
    let boundary_types = config.boundary_types();

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
        boundary_type: boundary_types,
        shrink_wrap_padding: config.shrink_wrap_padding,
        bounds_changed: false,
        volume: size[0] * size[1] * size[2],
        ghost_cutoff: 0.0,
    }
}

// ── Plugin ───────────────────────────────────────────────────────────────────

/// Registers [`Domain`] resource and periodic boundary condition systems.
pub struct DomainPlugin;

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
# Boundary type per axis: "periodic", "fixed", or "shrink-wrap"
boundary_x = "periodic"
boundary_y = "periodic"
boundary_z = "periodic"
# Padding for shrink-wrap boundaries [simulation units]. 0 = auto (use ghost cutoff).
# shrink_wrap_padding = 0.0"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<DomainConfig>(app, "domain");

        app.add_resource(Domain::new())
            .add_setup_system(domain_read_input, ScheduleSetupSet::Setup)
            .add_update_system(
                shrink_wrap.label("shrink_wrap").before("pbc"),
                ParticleSimScheduleSet::PreExchange,
            )
            .add_update_system(
                pbc.label("pbc").run_if(in_state(CommState::FullRebuild)),
                ParticleSimScheduleSet::PreExchange,
            );
    }
}

fn boundary_type_char(bt: BoundaryType) -> char {
    match bt {
        BoundaryType::Periodic => 'p',
        BoundaryType::Fixed => 'f',
        BoundaryType::ShrinkWrap => 's',
    }
}

/// Setup system: read `[domain]` config and initialize the [`Domain`] resource.
pub fn domain_read_input(
    config: Res<DomainConfig>,
    comm: Res<CommResource>,
    mut domain: ResMut<Domain>,
) {
    let boundary_types = config.boundary_types();

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

    *domain = decompose_domain(&config, &**comm);
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
    let any_shrink = domain.is_shrink_wrap(0) || domain.is_shrink_wrap(1) || domain.is_shrink_wrap(2);
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
        if !domain.is_shrink_wrap(d) {
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
/// Returns (new_pos, image_delta) where image_delta is +1/-1/0.
#[inline]
fn wrap_periodic(mut pos: f64, low: f64, size: f64) -> (f64, i32) {
    let high = low + size;
    if pos < low {
        pos += size;
        (pos, -1)
    } else if pos >= high {
        pos -= size;
        (pos, 1)
    } else {
        (pos, 0)
    }
}

/// Apply periodic boundary conditions: wrap positions on periodic axes,
/// remove out-of-bounds atoms on fixed/shrink-wrap axes.
///
/// Gated by `run_if(in_state(CommState::FullRebuild))` — only runs on full rebuild steps.
pub fn pbc(
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
    mut comm_state: ResMut<CurrentState<CommState>>,
) {
    let low = domain.boundaries_low;
    let high = domain.boundaries_high;
    let size = domain.size;
    let periodic = domain.periodic_flags();

    if periodic[0] && periodic[1] && periodic[2] {
        // Fast path: fully periodic, no removals possible (local atoms only, ghosts live outside box)
        for i in 0..atoms.nlocal as usize {
            for d in 0..3 {
                let (new_pos, delta) = wrap_periodic(atoms.pos[i][d], low[d], size[d]);
                atoms.pos[i][d] = new_pos;
                atoms.image[i][d] += delta;
            }
        }
    } else {
        // Slow path: non-periodic axes may require removal (local atoms only).
        // Shrink-wrap axes: atoms are always inside bounds (shrink_wrap ran first),
        // so the out-of-bounds check is a no-op — but it's harmless and correct.
        let nlocal_before = atoms.nlocal as usize;
        let mut removed = 0usize;
        'outer: for i in (0..atoms.nlocal as usize).rev() {
            for d in 0..3 {
                if periodic[d] {
                    let (new_pos, delta) = wrap_periodic(atoms.pos[i][d], low[d], size[d]);
                    atoms.pos[i][d] = new_pos;
                    atoms.image[i][d] += delta;
                } else if atoms.pos[i][d] < low[d] || atoms.pos[i][d] >= high[d] {
                    atoms.swap_remove(i);
                    registry.swap_remove_all(i);
                    removed += 1;
                    continue 'outer;
                }
            }
        }
        // Update nlocal and force full rebuild since sendlists are stale.
        if removed > 0 {
            atoms.nlocal = (nlocal_before - removed) as u32;
            comm_state.0 = CommState::FullRebuild;
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
            boundary_x: BoundaryType::Periodic,
            boundary_y: BoundaryType::Fixed,
            boundary_z: BoundaryType::Periodic,
            ..Default::default()
        };
        let comm = make_comm([1, 1, 1], [0, 0, 0]);
        let domain = decompose_domain(&config, &comm);

        assert_eq!(domain.boundaries_low, [0.0, 0.0, 0.0]);
        assert_eq!(domain.boundaries_high, [10.0, 5.0, 2.0]);
        assert_eq!(domain.sub_domain_low, [0.0, 0.0, 0.0]);
        assert_eq!(domain.sub_domain_high, [10.0, 5.0, 2.0]);
        assert!(domain.is_periodic(0));
        assert!(!domain.is_periodic(1));
        assert!(domain.is_periodic(2));
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
            ..Default::default()
        };
        // Simulate proc at position (1,0,0) in a 2x1x1 decomposition
        let comm = make_comm([2, 1, 1], [1, 0, 0]);
        let domain = decompose_domain(&config, &comm);

        assert!((domain.sub_domain_low[0] - 5.0).abs() < 1e-10);
        assert!((domain.sub_domain_high[0] - 10.0).abs() < 1e-10);
        assert!((domain.sub_length[0] - 5.0).abs() < 1e-10);
    }

    // ── Boundary type tests ────────────────────────────────────────────────

    #[test]
    fn boundary_types_array() {
        let config = DomainConfig {
            boundary_x: BoundaryType::Periodic,
            boundary_y: BoundaryType::Fixed,
            boundary_z: BoundaryType::ShrinkWrap,
            ..Default::default()
        };
        let types = config.boundary_types();
        assert_eq!(types, [BoundaryType::Periodic, BoundaryType::Fixed, BoundaryType::ShrinkWrap]);
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
            boundary_z: BoundaryType::ShrinkWrap,
            ..Default::default()
        };
        let comm = make_comm([1, 1, 1], [0, 0, 0]);
        let domain = decompose_domain(&config, &comm);

        assert!(domain.is_periodic(0));
        assert!(domain.is_periodic(1));
        assert!(!domain.is_periodic(2));
        assert!(!domain.is_shrink_wrap(0));
        assert!(!domain.is_shrink_wrap(1));
        assert!(domain.is_shrink_wrap(2));
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
            boundary_x = "periodic"
            boundary_y = "fixed"
            boundary_z = "shrink-wrap"
        "#;
        let config: DomainConfig = toml::from_str(toml_str).unwrap();
        let types = config.boundary_types();
        assert_eq!(types, [BoundaryType::Periodic, BoundaryType::Fixed, BoundaryType::ShrinkWrap]);
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
        domain.boundary_type = [BoundaryType::Periodic, BoundaryType::Periodic, BoundaryType::ShrinkWrap];
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
        domain.boundary_type = [BoundaryType::Periodic, BoundaryType::Periodic, BoundaryType::ShrinkWrap];
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
        domain.boundary_type = [BoundaryType::ShrinkWrap, BoundaryType::ShrinkWrap, BoundaryType::ShrinkWrap];
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
        domain.boundary_type = [BoundaryType::ShrinkWrap, BoundaryType::ShrinkWrap, BoundaryType::ShrinkWrap];
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
        domain.boundary_type = [BoundaryType::Periodic, BoundaryType::Periodic, BoundaryType::ShrinkWrap];
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

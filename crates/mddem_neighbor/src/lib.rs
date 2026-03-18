//! Neighbor list construction for particle simulations.
//!
//! This crate provides three neighbor-finding strategies, each with different
//! performance characteristics:
//!
//! | Strategy | Complexity | Best for |
//! |---|---|---|
//! | [`BruteForce`](NeighborStyle::BruteForce) | O(N²) | Tiny systems (< 100 atoms), debugging |
//! | [`SweepAndPrune`](NeighborStyle::SweepAndPrune) | O(N log N) | Small-to-medium systems without binning |
//! | [`Bin`](NeighborStyle::Bin) | O(N) expected | Production runs, large systems |
//!
//! All strategies produce a **half neighbor list** in CSR (Compressed Sparse Row)
//! format: each local atom `i` stores its neighbor indices `j > i` (or ghost atoms)
//! in a flat array, with offsets delimiting each atom's neighbors. Use
//! [`Neighbor::pairs()`] to iterate over `(i, j)` pairs efficiently.
//!
//! # Rebuild strategies
//!
//! Neighbor lists are rebuilt based on configurable criteria:
//!
//! - **Displacement-based** (`every = 0`): rebuilds when any atom moves more than
//!   `(skin_fraction - 1) * min_cutoff_radius` since the last build.
//! - **Periodic** (`every = N`): rebuilds every N steps.
//! - **Hybrid** (`every = N, check = true`): rebuilds every N steps OR on displacement,
//!   whichever comes first (like LAMMPS `neigh_modify every N check yes`).
//!
//! # Configuration
//!
//! Configure via the `[neighbor]` TOML section (see [`NeighborConfig`]).

use sim_app::prelude::*;
use sim_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

use mddem_core::{Atom, AtomDataRegistry, CommResource, Config, Domain, ScheduleSet, ScheduleSetupSet};

fn default_one_f64() -> f64 {
    1.0
}
fn default_zero_usize() -> usize {
    0
}
fn default_true() -> bool {
    true
}
fn default_sort_every() -> usize {
    1000
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[neighbor]` — neighbor list rebuild and binning settings.
pub struct NeighborConfig {
    /// Multiplier on pairwise cutoff for neighbor skin distance.
    #[serde(default = "default_one_f64")]
    pub skin_fraction: f64,
    /// Minimum bin size for bin-based neighbor lists.
    #[serde(default = "default_one_f64")]
    pub bin_size: f64,
    /// Rebuild every N steps (0 = displacement-based only)
    #[serde(default = "default_zero_usize")]
    pub every: usize,
    /// When true and every > 0, also check displacement threshold (like LAMMPS "check yes")
    #[serde(default = "default_true")]
    pub check: bool,
    /// Sort atoms by spatial bin every N steps for cache locality (0 = disabled).
    #[serde(default = "default_sort_every")]
    pub sort_every: usize,
    /// When true, PBC boundary crossings force a full neighbor rebuild.
    /// Required for DEM (contact forces are discontinuous; missed contacts cause energy spikes).
    #[serde(default)]
    pub rebuild_on_pbc_wrap: bool,
}

impl Default for NeighborConfig {
    fn default() -> Self {
        NeighborConfig {
            skin_fraction: 1.0,
            bin_size: 1.0,
            every: 0,
            check: true,
            sort_every: 1000,
            rebuild_on_pbc_wrap: false,
        }
    }
}

/// Algorithm used to build the neighbor list.
///
/// Choose based on system size and whether spatial binning overhead is worthwhile.
pub enum NeighborStyle {
    /// Check all N*(N-1)/2 pairs. O(N²) time, zero setup cost.
    /// Only suitable for very small systems or correctness testing.
    BruteForce,
    /// Sort atoms along the x-axis, then sweep to prune distant pairs.
    /// O(N log N) from the sort, with a linear sweep. No spatial grid needed.
    SweepAndPrune,
    /// Spatial binning with CSR (Compressed Sparse Row) layout and a precomputed
    /// stencil of neighbor cells. O(N) expected time for uniform distributions.
    /// Includes cache-friendly sorted position arrays and optional atom reordering.
    Bin,
}

/// Neighbor list state: pair lists, CSR indices, bin grid, and rebuild tracking.
///
/// The primary output is the CSR neighbor list stored in [`neighbor_offsets`](Self::neighbor_offsets)
/// and [`neighbor_indices`](Self::neighbor_indices). Use [`pairs()`](Self::pairs) to iterate
/// over `(i, j)` neighbor pairs.
pub struct Neighbor {
    /// Multiplier on pairwise cutoff: pair cutoff = `(r_i + r_j) * skin_fraction`.
    /// Values > 1.0 add a "skin" buffer to reduce rebuild frequency.
    pub skin_fraction: f64,
    /// Legacy pair list used by brute-force and sweep-and-prune (not used by bin strategy).
    pub neighbor_list: Vec<(usize, usize)>,
    /// CSR row offsets: `neighbor_offsets[i]..neighbor_offsets[i+1]` gives the range
    /// of neighbor indices for local atom `i`. Length = `nlocal + 1`.
    pub neighbor_offsets: Vec<u32>,
    /// CSR column indices: flat array of neighbor atom indices (local or ghost).
    pub neighbor_indices: Vec<u32>,
    /// Scratch space for sweep-and-prune: `(atom_index, x_position)` sorted by x.
    pub sweep_and_prune: Vec<(usize, f64)>,
    /// User-configured minimum bin size (may be increased to match cutoff).
    pub bin_min_size: f64,
    /// Actual bin dimensions in each axis `[bx, by, bz]`, computed from domain / bin count.
    pub bin_size: [f64; 3],
    /// Number of bins in each axis `[nx, ny, nz]` (including ghost layers).
    pub bin_count: [i32; 3],
    /// Saved atom positions from the last neighbor build, for displacement checking.
    pub last_build_pos: Vec<[f64; 3]>,
    /// Number of timesteps since the last neighbor list rebuild.
    pub steps_since_build: usize,
    /// Total atom count (local + ghost) at the last neighbor build.
    pub last_build_total: usize,
    /// Full simulation box dimensions `[Lx, Ly, Lz]` for minimum-image displacement checks.
    pub pbc_box: [f64; 3],
    /// Which axes have periodic boundary conditions.
    pub pbc_flags: [bool; 3],
    /// Full stencil: flat cell offsets `dx*ny*nz + dy*nz + dz` for all neighbor cells
    /// within cutoff distance (both forward and backward).
    pub bin_stencil: Vec<i32>,
    /// Forward-only stencil: cell offsets with `offset > 0`, used for half-neighbor-list
    /// construction to avoid counting each pair twice.
    pub bin_stencil_forward: Vec<i32>,
    /// Whether the self-cell (offset 0) passes the stencil distance test.
    pub bin_stencil_self: bool,
    /// Lower-left corner of the bin grid, offset by ghost layers from `sub_domain_low`.
    pub bin_origin: [f64; 3],
    /// Total number of bin cells: `nx * ny * nz` (including ghost layers).
    pub bin_total_cells: usize,
    /// Rebuild every N steps (0 = displacement-based only).
    pub every: usize,
    /// When true and `every > 0`, also check displacement threshold each step.
    pub check: bool,
    /// Communication cutoff for ghost atoms: `pair_cutoff + 2 * displacement_buffer`.
    pub ghost_cutoff: f64,
    /// Smallest cutoff radius among local atoms, cached at rebuild time for
    /// displacement threshold computation.
    pub cached_min_skin: f64,
    /// Reorder atoms by spatial bin every N steps for cache locality (0 = disabled).
    pub sort_every: usize,
    /// Steps elapsed since the last spatial sort.
    pub sort_counter: usize,
    /// Per-atom bin cell index (reused across rebuilds to avoid allocation).
    pub bin_atom_cell: Vec<u32>,
    /// Per-cell atom count, then reused as write cursor during CSR construction.
    pub bin_count_arr: Vec<u32>,
    /// CSR bin offsets: `bin_start[c]..bin_start[c+1]` gives sorted atom range for cell `c`.
    pub bin_start: Vec<u32>,
    /// Atoms sorted by bin cell (indices into the atom arrays).
    pub bin_sorted_atoms: Vec<u32>,
    /// Inverse of `bin_sorted_atoms`: position of atom `i` within `bin_sorted_atoms`.
    /// Used for self-cell skip optimization (start scanning after atom `i`).
    pub bin_atom_sorted_idx: Vec<u32>,
    /// Positions reordered by bin for cache-friendly inner loop access.
    pub bin_sorted_pos: Vec<[f64; 3]>,
    /// When all atoms share the same cutoff radius, cache `(2 * r * skin_fraction)²`
    /// to skip per-pair cutoff computation. `None` for polydisperse systems.
    pub cached_uniform_cutoff_sq: Option<f64>,
    /// Max pairwise cutoff from `neighbor_setup`, needed when recomputing bins
    /// after shrink-wrap domain changes.
    pub cached_max_cutoff: f64,
}

impl Default for Neighbor {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over `(i, j)` neighbor pairs from the CSR neighbor list.
///
/// Created by [`Neighbor::pairs()`]. Yields each pair exactly once, with `i` as a
/// local atom index and `j` as either local or ghost. Uses `unsafe` index access
/// for performance in the inner loop (validated by CSR construction invariants).
pub struct PairIter<'a> {
    offsets: &'a [u32],
    indices: &'a [u32],
    nlocal: usize,
    /// Current local atom index (row in the CSR).
    i: usize,
    /// Current position within `indices` for atom `i`'s neighbors.
    k: usize,
    /// End position within `indices` for atom `i`'s neighbors.
    end: usize,
}

impl<'a> Iterator for PairIter<'a> {
    type Item = (usize, usize);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.k >= self.end {
            self.i += 1;
            if self.i >= self.nlocal {
                return None;
            }
            // SAFETY: self.i < self.nlocal <= offsets.len() - 1, so self.i and self.i + 1 are in bounds.
            unsafe {
                self.k = *self.offsets.get_unchecked(self.i) as usize;
                self.end = *self.offsets.get_unchecked(self.i + 1) as usize;
            }
        }
        // SAFETY: self.k < self.end <= indices.len() (from CSR construction).
        let j = unsafe { *self.indices.get_unchecked(self.k) } as usize;
        self.k += 1;
        Some((self.i, j))
    }
}

impl Neighbor {
    /// Iterate over all (i, j) pairs from the CSR neighbor list.
    /// `nlocal` is the number of local atoms (only local atoms own neighbor lists).
    pub fn pairs(&self, nlocal: usize) -> PairIter<'_> {
        let end = if nlocal > 0 {
            self.neighbor_offsets[1] as usize
        } else {
            0
        };
        PairIter {
            offsets: &self.neighbor_offsets,
            indices: &self.neighbor_indices,
            nlocal,
            i: 0,
            k: if nlocal > 0 {
                self.neighbor_offsets[0] as usize
            } else {
                0
            },
            end,
        }
    }

    /// Creates a new `Neighbor` with default values.
    ///
    /// All arrays start empty; the bin grid and stencil are computed during
    /// [`neighbor_setup`] after the simulation domain and atom data are available.
    pub fn new() -> Self {
        Neighbor {
            skin_fraction: 1.0,
            neighbor_list: Vec::new(),
            neighbor_offsets: Vec::new(),
            neighbor_indices: Vec::new(),
            sweep_and_prune: Vec::new(),
            bin_min_size: 1.0,
            bin_size: [1.0; 3],
            bin_count: [1, 1, 1],
            last_build_pos: Vec::new(),
            steps_since_build: 0,
            last_build_total: 0,
            pbc_box: [0.0; 3],
            pbc_flags: [false; 3],
            bin_stencil: Vec::new(),
            bin_stencil_forward: Vec::new(),
            bin_stencil_self: false,
            bin_origin: [0.0; 3],
            bin_total_cells: 0,
            every: 0,
            check: true,
            ghost_cutoff: 0.0,
            cached_min_skin: f64::MAX,
            sort_every: 1000,
            sort_counter: 0,
            bin_atom_cell: Vec::new(),
            bin_count_arr: Vec::new(),
            bin_start: Vec::new(),
            bin_sorted_atoms: Vec::new(),
            bin_atom_sorted_idx: Vec::new(),
            bin_sorted_pos: Vec::new(),
            cached_uniform_cutoff_sq: None,
            cached_max_cutoff: 0.0,
        }
    }
}

/// Compute bin grid parameters (count, size, origin, stencil) from domain bounds and cutoff.
///
/// Shared helper used by both [`neighbor_setup`] (initial setup) and [`recompute_bins`]
/// (after shrink-wrap domain changes). Updates `bin_count`, `bin_size`, `bin_origin`,
/// `bin_total_cells`, stencil arrays, and PBC box dimensions on `neighbor`.
///
/// # Bin grid layout
///
/// The bin grid extends beyond the sub-domain by `sx`/`sy`/`sz` ghost layers on each
/// side, where `s = ceil(cutoff / bin_size)`. This ensures ghost atoms (which extend
/// up to `cutoff` beyond the sub-domain) are binned into valid cells.
///
/// Cell indexing is row-major: `cell = cx * ny * nz + cy * nz + cz`.
///
/// # Stencil construction
///
/// The stencil lists cell offsets `(dx, dy, dz)` whose **minimum possible distance**
/// to the origin cell is less than the cutoff. The minimum distance between two cells
/// offset by `d` bins is `max(0, |d| - 1) * bin_size` (since atoms can be anywhere
/// within their cell). Only cells passing this spherical distance test are included.
fn compute_bin_grid(neighbor: &mut Neighbor, domain: &Domain, comm_size: i32) {
    let max_cutoff = neighbor.cached_max_cutoff;

    // Multi-process: bins must be at most cutoff/2 so each sub-domain has enough bins.
    // Single-process: bins >= cutoff gives stencil range = 1 (fewer cells to check).
    let required_bin = if comm_size > 1 { max_cutoff * 0.5 } else { max_cutoff };
    let min_bin = neighbor.bin_min_size.max(required_bin);

    // Compute number of interior bins per axis (at least 1), then actual bin sizes.
    let xi = (domain.sub_length[0] / min_bin).floor().max(1.0) as i32;
    let yi = (domain.sub_length[1] / min_bin).floor().max(1.0) as i32;
    let zi = (domain.sub_length[2] / min_bin).floor().max(1.0) as i32;

    let actual_bin_size = [
        domain.sub_length[0] / xi as f64,
        domain.sub_length[1] / yi as f64,
        domain.sub_length[2] / zi as f64,
    ];

    // Ghost layers per side: enough bins to cover the cutoff distance.
    let sx = (max_cutoff / actual_bin_size[0]).ceil() as i32;
    let sy = (max_cutoff / actual_bin_size[1]).ceil() as i32;
    let sz = (max_cutoff / actual_bin_size[2]).ceil() as i32;

    // Total bins = interior + 2 * ghost layers per axis.
    let nx = xi + 2 * sx;
    let ny = yi + 2 * sy;
    let nz = zi + 2 * sz;
    neighbor.bin_count = [nx, ny, nz];
    neighbor.bin_size = actual_bin_size;

    let total_cells = (nx * ny * nz) as usize;
    neighbor.bin_total_cells = total_cells;

    // Bin origin = sub-domain corner shifted left by ghost layers.
    neighbor.bin_origin = [
        domain.sub_domain_low[0] - actual_bin_size[0] * sx as f64,
        domain.sub_domain_low[1] - actual_bin_size[1] * sy as f64,
        domain.sub_domain_low[2] - actual_bin_size[2] * sz as f64,
    ];

    // Precompute stencil offsets — only include cells whose minimum distance < cutoff.
    // Minimum distance between cells offset by d bins = max(0, |d|-1) * bin_size,
    // because atoms within adjacent cells (|d|=1) can be arbitrarily close.
    let cutoff_sq = max_cutoff * max_cutoff;
    neighbor.bin_stencil.clear();
    neighbor.bin_stencil_forward.clear();
    neighbor.bin_stencil_self = false;
    for dx in -sx..=sx {
        for dy in -sy..=sy {
            for dz in -sz..=sz {
                let min_dx = (dx.unsigned_abs().saturating_sub(1)) as f64 * actual_bin_size[0];
                let min_dy = (dy.unsigned_abs().saturating_sub(1)) as f64 * actual_bin_size[1];
                let min_dz = (dz.unsigned_abs().saturating_sub(1)) as f64 * actual_bin_size[2];
                if min_dx * min_dx + min_dy * min_dy + min_dz * min_dz < cutoff_sq {
                    // Row-major cell offset for 3D -> 1D indexing.
                    let offset = dx * ny * nz + dy * nz + dz;
                    neighbor.bin_stencil.push(offset);
                    if offset > 0 {
                        neighbor.bin_stencil_forward.push(offset);
                    } else if offset == 0 {
                        neighbor.bin_stencil_self = true;
                    }
                }
            }
        }
    }

    // Cache full box dimensions for minimum-image displacement checks.
    neighbor.pbc_box = [
        domain.boundaries_high[0] - domain.boundaries_low[0],
        domain.boundaries_high[1] - domain.boundaries_low[1],
        domain.boundaries_high[2] - domain.boundaries_low[2],
    ];
}

/// Recompute bin grid parameters after domain bounds change (e.g., shrink-wrap).
fn recompute_bins(neighbor: &mut Neighbor, domain: &Domain, comm_size: i32) {
    if neighbor.cached_max_cutoff <= 0.0 {
        return; // not yet initialized
    }
    compute_bin_grid(neighbor, domain, comm_size);
}

/// Plugin that registers neighbor list construction and rebuild systems.
///
/// Add this plugin to your [`App`] to enable neighbor list building. The chosen
/// [`NeighborStyle`] determines which build algorithm runs each timestep.
///
/// # Systems registered
///
/// - **Setup**: [`neighbor_read_input`] (reads `[neighbor]` config) and
///   [`neighbor_setup`] (computes bin grid, ghost cutoff).
/// - **Update**: [`decide_rebuild`] (displacement check), plus the selected
///   neighbor build system ([`brute_force_neighbor_list`],
///   [`sweep_and_prune_neighbor_list`], or [`bin_neighbor_list`]).
/// - **Bin only**: [`sort_atoms_by_bin`] for cache-locality reordering.
pub struct NeighborPlugin {
    /// Which neighbor-finding algorithm to use.
    pub style: NeighborStyle,
}

impl Plugin for NeighborPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[neighbor]
# Skin fraction multiplier for neighbor list cutoff
skin_fraction = 1.0
# Bin size for bin-based neighbor list
bin_size = 1.0
# Rebuild every N steps (0 = displacement-based only)
every = 0
# Also check displacement when every > 0 (like LAMMPS "check yes")
check = true
# Sort atoms by spatial bin every N steps for cache locality (0 = disabled)
sort_every = 1000"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<NeighborConfig>(app, "neighbor");

        app.add_resource(Neighbor::new())
            .add_setup_system(neighbor_read_input, ScheduleSetupSet::Setup)
            .add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup)
            .add_update_system(
                decide_rebuild.label("decide_rebuild").before(mddem_core::remove_ghost_atoms),
                ScheduleSet::PostInitialIntegration,
            );
        match self.style {
            NeighborStyle::BruteForce => {
                app.add_update_system(brute_force_neighbor_list, ScheduleSet::Neighbor);
            }
            NeighborStyle::SweepAndPrune => {
                app.add_update_system(sweep_and_prune_neighbor_list, ScheduleSet::Neighbor);
            }
            NeighborStyle::Bin => {
                app.add_update_system(
                    sort_atoms_by_bin
                        .label("sort_atoms")
                        .before(mddem_core::borders),
                    ScheduleSet::PreNeighbor,
                );
                app.add_update_system(bin_neighbor_list, ScheduleSet::Neighbor);
            }
        }
    }
}

/// Setup system: reads `[neighbor]` config values into the [`Neighbor`] resource.
///
/// Runs at [`ScheduleSetupSet::Setup`]. Prints the rebuild strategy on rank 0.
pub fn neighbor_read_input(
    config: Res<NeighborConfig>,
    mut neighbor: ResMut<Neighbor>,
    comm: Res<CommResource>,
) {
    neighbor.skin_fraction = config.skin_fraction;
    neighbor.bin_min_size = config.bin_size;
    neighbor.every = config.every;
    neighbor.check = config.check;
    neighbor.sort_every = config.sort_every;
    if comm.rank() == 0 {
        let rebuild_str = if config.every == 0 {
            "displacement".to_string()
        } else if config.check {
            format!("every {} + check", config.every)
        } else {
            format!("every {}", config.every)
        };
        println!(
            "Neighbor: skin_fraction={} bin_size={} rebuild={}",
            config.skin_fraction, config.bin_size, rebuild_str
        );
    }
}

/// Setup system: computes bin grid, ghost cutoff, and stencil from atom cutoff radii.
///
/// Runs at [`ScheduleSetupSet::PostSetup`] after atoms are created. Determines:
/// - `max_cutoff = 2 * max_skin * skin_fraction` (largest pairwise neighbor distance)
/// - `ghost_cutoff = max_cutoff + displacement_buffer` (communication distance for ghosts)
/// - Bin grid dimensions, stencil offsets, and PBC flags
pub fn neighbor_setup(config: Res<NeighborConfig>, mut neighbor: ResMut<Neighbor>, mut domain: ResMut<Domain>, mut atoms: ResMut<Atom>, comm: Res<CommResource>) {
    // Compute max neighbor cutoff = (skin_i + skin_j) * skin_fraction = 2 * max_skin * skin_fraction
    // Use global reduction: at PostSetup, atoms may only be on rank 0 (before exchange).
    let local_max_skin = atoms.cutoff_radius.iter().cloned().fold(0.0f64, f64::max);
    let max_skin = -comm.all_reduce_min_f64(-local_max_skin); // global max via negated min
    // When no particles exist yet (e.g. rate-based insertion), fall back to bin_size
    // so ghost_cutoff is sensible. The cutoff will be updated on first neighbor rebuild.
    let max_cutoff = if max_skin > 0.0 {
        2.0 * max_skin * neighbor.skin_fraction
    } else {
        neighbor.bin_min_size
    };
    // Add displacement buffer to ghost_cutoff so atoms don't drift in/out of the
    // ghost zone between neighbor rebuilds. Without this padding, ghost count
    // fluctuates every step, forcing unnecessary neighbor rebuilds.
    // Max per-atom displacement before rebuild = (skin_fraction - 1) * min_skin.
    // Two atoms can each move this far, so buffer = 2 * displacement.
    let local_min_skin = atoms.cutoff_radius.iter().cloned().fold(f64::MAX, f64::min);
    let min_skin = comm.all_reduce_min_f64(local_min_skin);
    // Guard against f64::MAX when cutoff_radius is empty (rate-based insertion)
    let displacement_buffer = if min_skin < f64::MAX * 0.5 {
        (neighbor.skin_fraction - 1.0) * min_skin
    } else {
        0.0
    };
    let ghost_cut = max_cutoff + 2.0 * displacement_buffer;
    neighbor.ghost_cutoff = ghost_cut;
    neighbor.cached_max_cutoff = max_cutoff;
    domain.ghost_cutoff = ghost_cut;
    atoms.rebuild_on_pbc_wrap = config.rebuild_on_pbc_wrap;
    if comm.rank() == 0 {
        println!("Neighbor: ghost_cutoff={:.4} (pair_cutoff={:.4} + buffer={:.4})",
            ghost_cut, max_cutoff, 2.0 * displacement_buffer);
    }
    // Single-process: bin_size >= cutoff gives stencil range=1, keeping bin_start
    // small enough (< 64KB) for L1 cache and only 13 forward stencil cells.
    // Multi-process: use cutoff/2 so subdomains have enough bins for correct stencil.
    let required_bin = if comm.size() > 1 { max_cutoff * 0.5 } else { max_cutoff };
    if neighbor.bin_min_size < required_bin {
        neighbor.bin_min_size = required_bin;
    }

    let min_bin = neighbor.bin_min_size;
    if (domain.sub_length[0] / min_bin < 1.0
        || domain.sub_length[1] / min_bin < 1.0
        || domain.sub_length[2] / min_bin < 1.0)
        && comm.rank() == 0
    {
        println!("WARNING: subdomain smaller than bin_size in at least one dimension, clamping to 1 bin");
    }

    // Compute bin grid, stencil, and PBC box using shared helper
    compute_bin_grid(&mut neighbor, &domain, comm.size());
    neighbor.pbc_flags = domain.is_periodic;

    if comm.rank() == 0 {
        println!(
            "Neighbor: bins {}x{}x{} (with ghost layers), {} forward stencil cells",
            neighbor.bin_count[0], neighbor.bin_count[1], neighbor.bin_count[2],
            neighbor.bin_stencil_forward.len()
        );
    }
}

/// Helper: build merged CSR neighbor list from neighbor_list pairs
fn build_csr_from_pairs(neighbor: &mut Neighbor, nlocal: usize) {
    let prev = neighbor.neighbor_indices.len();
    neighbor.neighbor_offsets.clear();
    neighbor.neighbor_indices.clear();
    neighbor.neighbor_indices.reserve(prev + prev / 4);

    // Sort pairs by i for CSR construction
    let mut pairs = neighbor.neighbor_list.clone();
    pairs.sort_unstable_by_key(|&(i, _)| i);

    let mut pair_idx = 0;
    for i in 0..nlocal {
        neighbor.neighbor_offsets.push(neighbor.neighbor_indices.len() as u32);
        while pair_idx < pairs.len() && pairs[pair_idx].0 == i {
            let j = pairs[pair_idx].1;
            neighbor.neighbor_indices.push(j as u32);
            pair_idx += 1;
        }
    }
    neighbor.neighbor_offsets.push(neighbor.neighbor_indices.len() as u32);
}

/// Helper: save current positions for displacement-based rebuild check.
/// Saves local atom positions and all atom tags (including ghosts) for identity tracking.
fn save_build_positions(atoms: &Atom, neighbor: &mut Neighbor) {
    let nlocal = atoms.nlocal as usize;
    neighbor.last_build_pos.resize(nlocal, [0.0; 3]);
    neighbor.last_build_pos[..nlocal].copy_from_slice(&atoms.pos[..nlocal]);
    neighbor.last_build_total = atoms.len();
    neighbor.steps_since_build = 0;
    let (min_skin, max_skin) = atoms.cutoff_radius[..nlocal]
        .iter()
        .fold((f64::MAX, f64::MIN), |(mn, mx), &s| (mn.min(s), mx.max(s)));
    neighbor.cached_min_skin = min_skin;
    if (max_skin - min_skin).abs() < 1e-15 {
        let cutoff = 2.0 * min_skin * neighbor.skin_fraction;
        neighbor.cached_uniform_cutoff_sq = Some(cutoff * cutoff);
    } else {
        neighbor.cached_uniform_cutoff_sq = None;
    }
}

/// Helper: check displacement threshold
fn displacement_exceeded(atoms: &Atom, neighbor: &Neighbor) -> bool {
    let min_r = neighbor.cached_min_skin;
    // Per-atom displacement budget: the pair skin margin is
    //   (skin_fraction - 1) * (r_i + r_j).
    // Each atom's share is half:
    //   (skin_fraction - 1) * (r_i + r_j) / 2.
    // Using min_r as worst case for each radius:
    //   threshold = (skin_fraction - 1) * min_r
    let per_atom_skin = (neighbor.skin_fraction - 1.0) * min_r;
    let threshold_sq = per_atom_skin * per_atom_skin;
    // Minimum-image convention for periodic axes: when an atom wraps across a
    // periodic boundary, the raw displacement is ~box_size, but the physical
    // displacement is small. forward_comm recomputes per-atom periodic offsets
    // so ghost positions stay correct after PBC wrapping.
    let [bx, by, bz] = neighbor.pbc_box;
    let [px, py, pz] = neighbor.pbc_flags;
    let hbx = bx * 0.5;
    let hby = by * 0.5;
    let hbz = bz * 0.5;
    for idx in 0..neighbor.last_build_pos.len() {
        let mut dx = atoms.pos[idx][0] - neighbor.last_build_pos[idx][0];
        let mut dy = atoms.pos[idx][1] - neighbor.last_build_pos[idx][1];
        let mut dz = atoms.pos[idx][2] - neighbor.last_build_pos[idx][2];
        if px {
            if dx > hbx { dx -= bx; } else if dx < -hbx { dx += bx; }
        }
        if py {
            if dy > hby { dy -= by; } else if dy < -hby { dy += by; }
        }
        if pz {
            if dz > hbz { dz -= bz; } else if dz < -hbz { dz += bz; }
        }
        if dx * dx + dy * dy + dz * dz > threshold_sq {
            return true;
        }
    }
    false
}

/// Helper: check if a rebuild is needed based on atom count change,
/// step count, and displacement since last build.
fn needs_rebuild(atoms: &Atom, neighbor: &Neighbor) -> bool {
    let nlocal = atoms.nlocal as usize;
    // Always rebuild on first call or local atom count change
    if neighbor.last_build_pos.len() != nlocal || nlocal == 0 {
        return true;
    }
    // If communicate_only is false, a full rebuild was requested (first step,
    // or decide_rebuild detected displacement exceeded threshold).
    if !atoms.communicate_only {
        return true;
    }

    if neighbor.every == 0 {
        // Displacement-based only
        displacement_exceeded(atoms, neighbor)
    } else if neighbor.check {
        // Every N steps OR displacement exceeded
        neighbor.steps_since_build >= neighbor.every || displacement_exceeded(atoms, neighbor)
    } else {
        // Every N steps only (no displacement check)
        neighbor.steps_since_build >= neighbor.every
    }
}

/// Runs at PostInitialIntegration before remove_ghost_atoms.
/// When communicate_only is true (neighbor list still valid), checks displacement
/// to decide if a full rebuild is needed this step. If so, sets communicate_only = false
/// so that remove_ghost_atoms / exchange / full borders all run.
///
/// Uses all_reduce to ensure ALL ranks agree — if any rank needs a rebuild,
/// all ranks do the full rebuild (required for MPI send/recv pattern matching).
pub fn decide_rebuild(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    comm: Res<CommResource>,
    domain: Res<Domain>,
) {
    if !atoms.communicate_only {
        return; // already going to do full rebuild
    }
    let mut local_needs = if needs_rebuild(&atoms, &neighbor) { 1.0 } else { 0.0 };
    // When rebuild_on_pbc_wrap is set (DEM), detect any atom about to cross a
    // periodic boundary. After initial_integration moved atoms but before pbc()
    // wraps them, atoms outside the box are exactly the ones that will wrap.
    // Stale ghost placement after PBC wrap causes missed contacts in DEM.
    if local_needs == 0.0 && atoms.rebuild_on_pbc_wrap {
        let low = domain.boundaries_low;
        let high = domain.boundaries_high;
        for i in 0..atoms.nlocal as usize {
            if atoms.pos[i][0] < low[0] || atoms.pos[i][0] >= high[0]
                || atoms.pos[i][1] < low[1] || atoms.pos[i][1] >= high[1]
                || atoms.pos[i][2] < low[2] || atoms.pos[i][2] >= high[2]
            {
                local_needs = 1.0;
                break;
            }
        }
    }
    // Any rank needing rebuild forces all ranks to rebuild
    let global_needs = comm.all_reduce_sum_f64(local_needs);
    if global_needs > 0.0 {
        atoms.communicate_only = false;
    }
}

/// Sweep-and-prune neighbor list builder. O(N log N) from sorting atoms by x-coordinate.
///
/// Sorts all atoms by x-position, then sweeps forward: for each atom `i`, checks
/// atoms `j > i` in sorted order until the x-gap exceeds the cutoff, pruning
/// the search space. Full 3D distance is checked for remaining candidates.
///
/// Produces both a legacy pair list and a CSR neighbor list.
pub fn sweep_and_prune_neighbor_list(
    mut atoms: ResMut<Atom>,
    mut neighbor: ResMut<Neighbor>,
    _domain: Res<Domain>,
    _comm: Res<CommResource>,
) {
    if !needs_rebuild(&atoms, &neighbor) {
        neighbor.steps_since_build += 1;
        atoms.communicate_only = true;
        return;
    }
    atoms.communicate_only = true;

    save_build_positions(&atoms, &mut neighbor);
    neighbor.sweep_and_prune.clear();
    neighbor.neighbor_list.clear();

    for j in 0..atoms.len() {
        neighbor.sweep_and_prune.push((j, atoms.pos[j][0]));
    }
    neighbor
        .sweep_and_prune
        .sort_by(|a, b| a.1.partial_cmp(&b.1).expect("NaN in atom x-position during sweep-and-prune sort"));

    let skin_fraction = neighbor.skin_fraction;
    for i in 0..neighbor.sweep_and_prune.len() {
        let index = neighbor.sweep_and_prune[i].0;
        let px = atoms.pos[index][0];
        let py = atoms.pos[index][1];
        let pz = atoms.pos[index][2];
        let r = atoms.cutoff_radius[index];
        for j in (i + 1)..neighbor.sweep_and_prune.len() {
            if (neighbor.sweep_and_prune[j].1 - px) > (r * 2.0 * skin_fraction) {
                break;
            }
            let index2 = neighbor.sweep_and_prune[j].0;
            if atoms.tag[index] == atoms.tag[index2]
                || (atoms.is_ghost[index] && atoms.is_ghost[index2])
            {
                continue;
            }
            let r2 = atoms.cutoff_radius[index2];
            let dx = atoms.pos[index2][0] - px;
            let dy = atoms.pos[index2][1] - py;
            let dz = atoms.pos[index2][2] - pz;
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            if distance < (r + r2) * skin_fraction {
                neighbor.neighbor_list.push((index, index2));
            }
        }
    }
    let nlocal = atoms.nlocal as usize;
    build_csr_from_pairs(&mut neighbor, nlocal);
}

/// Brute-force neighbor list builder. O(N²) — checks all local-vs-all pairs.
///
/// Simple reference implementation: iterates over all `(i, j)` pairs with `i < j`,
/// skipping self-interactions (same tag). Suitable only for small systems or testing.
/// Does not use displacement-based rebuild skipping.
pub fn brute_force_neighbor_list(atoms: Res<Atom>, mut neighbor: ResMut<Neighbor>) {
    neighbor.neighbor_list.clear();
    let nlocal = atoms.len() - atoms.nghost as usize;
    for i in 0..nlocal {
        for j in (i + 1)..atoms.len() {
            if atoms.tag[i] == atoms.tag[j] {
                continue;
            }
            let dx = atoms.pos[j][0] - atoms.pos[i][0];
            let dy = atoms.pos[j][1] - atoms.pos[i][1];
            let dz = atoms.pos[j][2] - atoms.pos[i][2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            if distance < (atoms.cutoff_radius[i] + atoms.cutoff_radius[j]) * neighbor.skin_fraction {
                neighbor.neighbor_list.push((i, j));
            }
        }
    }
    build_csr_from_pairs(&mut neighbor, nlocal);
}

/// Reorder local atoms by spatial bin for improved cache locality.
///
/// Runs at [`ScheduleSet::PreNeighbor`] (before ghost communication). Atoms are
/// sorted by their bin cell index so that spatially nearby atoms are contiguous
/// in memory, improving cache hit rates during neighbor list construction and
/// force computation.
///
/// Sorting is triggered either periodically (every `sort_every` steps) or when
/// a neighbor rebuild is needed. The permutation is applied to all atom arrays
/// (via [`AtomDataRegistry`]) and ghost `origin_index` values are updated.
pub fn sort_atoms_by_bin(mut atoms: ResMut<Atom>, mut neighbor: ResMut<Neighbor>, comm: Res<CommResource>, registry: Res<AtomDataRegistry>, mut domain: ResMut<Domain>) {
    // Recompute bin grid if domain bounds changed (e.g., shrink-wrap).
    // Must happen before sorting since sorting depends on bin parameters.
    if domain.bounds_changed {
        recompute_bins(&mut neighbor, &domain, comm.size());
        domain.bounds_changed = false;
        atoms.communicate_only = false;
    }
    // Increment sort_counter first so it stays synchronized across all MPI ranks,
    // even if some ranks skip sorting due to nlocal == 0 or other conditions.
    neighbor.sort_counter += 1;
    let periodic_sort = neighbor.sort_every > 0 && neighbor.sort_counter >= neighbor.sort_every;
    if periodic_sort {
        neighbor.sort_counter = 0;
    }

    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 || neighbor.bin_total_cells == 0 || nlocal > atoms.pos.len() {
        // Even with nothing to sort locally, must participate in the all_reduce
        // when periodic_sort triggers so all MPI ranks stay synchronized.
        if periodic_sort {
            let global_did_sort = comm.all_reduce_sum_f64(0.0);
            if global_did_sort > 0.0 {
                neighbor.last_build_pos.clear();
                atoms.communicate_only = false;
            }
        }
        return;
    }

    let rebuild_needed = needs_rebuild(&atoms, &neighbor);
    if !periodic_sort && !rebuild_needed {
        return;
    }

    let inv_bsx = 1.0 / neighbor.bin_size[0];
    let inv_bsy = 1.0 / neighbor.bin_size[1];
    let inv_bsz = 1.0 / neighbor.bin_size[2];
    let ny = neighbor.bin_count[1];
    let nz = neighbor.bin_count[2];
    let nx = neighbor.bin_count[0];

    let mut indices: Vec<(u32, usize)> = (0..nlocal)
        .map(|i| {
            let cx = ((atoms.pos[i][0] - neighbor.bin_origin[0]) * inv_bsx).floor() as i32;
            let cy = ((atoms.pos[i][1] - neighbor.bin_origin[1]) * inv_bsy).floor() as i32;
            let cz = ((atoms.pos[i][2] - neighbor.bin_origin[2]) * inv_bsz).floor() as i32;
            let cx = cx.clamp(0, nx - 1);
            let cy = cy.clamp(0, ny - 1);
            let cz = cz.clamp(0, nz - 1);
            ((cx * ny * nz + cy * nz + cz) as u32, i)
        })
        .collect();

    indices.sort_unstable_by_key(|&(bin, _)| bin);

    let perm: Vec<usize> = indices.iter().map(|&(_, i)| i).collect();

    // Skip permutation if atoms are already in order
    let already_sorted = perm.iter().enumerate().all(|(i, &p)| p == i);

    if !already_sorted {
        atoms.apply_permutation(&perm, nlocal);
        registry.apply_permutation_all(&perm, nlocal);

        // Apply the same permutation to last_build_pos so displacement checks
        // compare the correct atom's current position against its saved position.
        if neighbor.last_build_pos.len() >= nlocal {
            let old = neighbor.last_build_pos.clone();
            for (new_i, &old_i) in perm.iter().enumerate() {
                neighbor.last_build_pos[new_i] = old[old_i];
            }
        }

        // Update ghost origin_index: after sort, local atom at old_i is now at new_i.
        // Build inverse permutation: inv_perm[old_i] = new_i.
        // Only remap origin_indices pointing to local atoms (< nlocal).
        // Ghost-of-ghost origin_indices point to other ghosts which aren't sorted.
        let nghost = atoms.nghost as usize;
        if nghost > 0 {
            let mut inv_perm = vec![0u32; nlocal];
            for (new_i, &old_i) in perm.iter().enumerate() {
                inv_perm[old_i] = new_i as u32;
            }
            for gi in nlocal..(nlocal + nghost) {
                let old_origin = atoms.origin_index[gi] as usize;
                if old_origin < nlocal {
                    atoms.origin_index[gi] = inv_perm[old_origin] as i32;
                }
            }
        }
    }

    // After permutation, swap_data.send_indices are stale (they reference pre-sort
    // local indices). Force a full borders rebuild so new swap_data is generated.
    // When periodic_sort triggered, all ranks entered this function (they share the
    // same sort_counter). Use all_reduce to synchronize: if ANY rank permuted, ALL
    // ranks must do a full borders rebuild (borders uses collective MPI communication
    // that requires all ranks to follow the same code path).
    // The all_reduce MUST happen after all early returns so every rank participates.
    if periodic_sort {
        let local_did_sort = if already_sorted { 0.0 } else { 1.0 };
        let global_did_sort = comm.all_reduce_sum_f64(local_did_sort);
        if global_did_sort > 0.0 {
            neighbor.last_build_pos.clear();
            atoms.communicate_only = false;
        }
    }
}

/// Bin-based neighbor list builder. O(N) expected time for uniform particle distributions.
///
/// Uses a spatial bin grid with a precomputed stencil of neighbor cells. The algorithm:
///
/// 1. **Assign** each atom to a bin cell based on its position (counting sort).
/// 2. **Build** CSR bin offsets so `bin_start[c]..bin_start[c+1]` gives atoms in cell `c`.
/// 3. **Scan** each local atom's self-cell (j > i only) and forward stencil cells to find
///    neighbors within cutoff, producing a half neighbor list in CSR format.
///
/// Two code paths are used: a **fast path** when all atoms share the same cutoff radius
/// (skips per-pair cutoff computation), and a **slow path** for polydisperse systems.
///
/// All inner loops use `unsafe` unchecked indexing for performance; safety invariants
/// are documented inline and validated by the CSR construction logic.
pub fn bin_neighbor_list(
    mut atoms: ResMut<Atom>,
    mut neighbor: ResMut<Neighbor>,
    mut domain: ResMut<Domain>,
    comm: Res<CommResource>,
) {
    // Recompute bin grid if domain bounds changed (e.g., shrink-wrap).
    // This is a fallback — sort_atoms_by_bin handles this for the Bin style,
    // but BruteForce/SweepAndPrune callers won't have sort_atoms_by_bin.
    if domain.bounds_changed {
        recompute_bins(&mut neighbor, &domain, comm.size());
        domain.bounds_changed = false;
        atoms.communicate_only = false;
    }

    let nlocal = atoms.nlocal as usize;
    let total = atoms.len();
    if !needs_rebuild(&atoms, &neighbor) {
        neighbor.steps_since_build += 1;
        atoms.communicate_only = true;
        return;
    }
    atoms.communicate_only = true;

    save_build_positions(&atoms, &mut neighbor);

    let ny = neighbor.bin_count[1];
    let nz = neighbor.bin_count[2];
    let nx = neighbor.bin_count[0];
    let total_cells = neighbor.bin_total_cells;
    let inv_bsx = 1.0 / neighbor.bin_size[0];
    let inv_bsy = 1.0 / neighbor.bin_size[1];
    let inv_bsz = 1.0 / neighbor.bin_size[2];
    let bin_ox = neighbor.bin_origin[0];
    let bin_oy = neighbor.bin_origin[1];
    let bin_oz = neighbor.bin_origin[2];

    // Step 1: Assign each atom to a bin cell via floor((pos - origin) / bin_size).
    // Reuse persistent arrays (taken via mem::take, returned at end) to avoid allocation.
    let mut atom_cell = std::mem::take(&mut neighbor.bin_atom_cell);
    atom_cell.clear();
    atom_cell.resize(total, 0u32);
    let mut bin_count_arr = std::mem::take(&mut neighbor.bin_count_arr);
    bin_count_arr.clear();
    bin_count_arr.resize(total_cells, 0u32);

    // SAFETY: i < total = atoms.len(), cell is clamped to 0..total_cells by clamp on cx/cy/cz.
    for i in 0..total {
        let pi = unsafe { atoms.pos.get_unchecked(i) };
        let cx = ((pi[0] - bin_ox) * inv_bsx).floor() as i32;
        let cy = ((pi[1] - bin_oy) * inv_bsy).floor() as i32;
        let cz = ((pi[2] - bin_oz) * inv_bsz).floor() as i32;
        let cx = cx.clamp(0, nx - 1);
        let cy = cy.clamp(0, ny - 1);
        let cz = cz.clamp(0, nz - 1);
        let cell = (cx * ny * nz + cy * nz + cz) as u32;
        unsafe {
            *atom_cell.get_unchecked_mut(i) = cell;
            *bin_count_arr.get_unchecked_mut(cell as usize) += 1;
        }
    }

    // Step 2: Build CSR bin offsets (reuse persistent array)
    let mut bin_start = std::mem::take(&mut neighbor.bin_start);
    bin_start.clear();
    bin_start.resize(total_cells + 1, 0u32);
    for c in 0..total_cells {
        bin_start[c + 1] = bin_start[c] + bin_count_arr[c];
    }

    // Place atoms into sorted order by bin — reuse bin_count_arr as write cursor
    // Also record each atom's position in sorted_atoms for self-cell skip optimization
    let mut sorted_atoms = std::mem::take(&mut neighbor.bin_sorted_atoms);
    sorted_atoms.clear();
    sorted_atoms.resize(total, 0u32);
    let mut atom_sorted_idx = std::mem::take(&mut neighbor.bin_atom_sorted_idx);
    atom_sorted_idx.clear();
    atom_sorted_idx.resize(total, 0u32);
    bin_count_arr[..total_cells].copy_from_slice(&bin_start[..total_cells]);
    for i in 0..total {
        let c = atom_cell[i] as usize;
        let pos = bin_count_arr[c];
        sorted_atoms[pos as usize] = i as u32;
        atom_sorted_idx[i] = pos;
        bin_count_arr[c] = pos + 1;
    }

    // Step 2b: Build sorted position cache for cache-friendly inner loop access
    let mut sorted_pos = std::mem::take(&mut neighbor.bin_sorted_pos);
    sorted_pos.clear();
    sorted_pos.resize(total, [0.0; 3]);
    for m in 0..total {
        // SAFETY: sorted_atoms[m] was populated from 0..total, so < atoms.pos.len().
        sorted_pos[m] = unsafe { *atoms.pos.get_unchecked(*sorted_atoms.get_unchecked(m) as usize) };
    }

    // Step 3: Build CSR neighbor lists using forward stencil.
    // Each pair is found exactly once: self cell uses j > i dedup,
    // forward cells have positive offset so each pair appears once.
    let has_self = neighbor.bin_stencil_self;
    let skin_fraction = neighbor.skin_fraction;
    let uniform_cutoff_sq = neighbor.cached_uniform_cutoff_sq;
    // Take stencil to avoid borrow conflict with neighbor_indices
    let stencil_forward = std::mem::take(&mut neighbor.bin_stencil_forward);

    let prev_count = neighbor.neighbor_indices.len();
    // Work with local vecs to avoid borrow conflicts through ResMut
    let mut offsets = std::mem::take(&mut neighbor.neighbor_offsets);
    let mut indices = std::mem::take(&mut neighbor.neighbor_indices);
    offsets.clear();
    indices.clear();
    offsets.reserve(nlocal + 1);
    // Pre-allocate indices buffer generously; use unchecked writes with a counter
    let indices_cap = (prev_count + prev_count / 4).max(nlocal * 40);
    indices.reserve(indices_cap);
    let mut nidx: usize = 0;

    // Macro to grow indices if needed, then write unchecked
    macro_rules! push_index {
        ($val:expr) => {
            if nidx >= indices.capacity() {
                // SAFETY: nidx == current logical length
                unsafe { indices.set_len(nidx) };
                indices.reserve(nidx / 2 + 256);
                // indices_ptr is stale after realloc — but we shadow it below
            }
            // SAFETY: nidx < capacity after potential growth
            unsafe { *indices.as_mut_ptr().add(nidx) = $val };
            nidx += 1;
        };
    }

    // SAFETY invariants for all inner loops below:
    // - m ranges bin_start[c]..bin_start[c+1], both < total (prefix sum of counts summing to total)
    // - sorted_atoms[m] < total (populated from 0..total)
    // - sorted_pos[m] valid for same range as sorted_atoms
    // - c = my_cell + offset, where my_cell is clamped to valid bin range and offset is from stencil
    // - bin_start has total_cells + 1 entries; c and c+1 are within bounds by stencil construction
    // - Self-cell: start from atom_sorted_idx[i]+1 to skip all atoms at or before i in the bin.
    //   Within the same bin, local atoms appear in index order (counting sort preserves insertion
    //   order, and sort_atoms_by_bin pre-sorts locals by bin). Ghosts have index >= nlocal > i.
    if let Some(cutoff_sq) = uniform_cutoff_sq {
        // Fast path: all atoms have the same skin — skip per-pair skin load
        for i in 0..nlocal {
            offsets.push(nidx as u32);
            let my_cell = unsafe { *atom_cell.get_unchecked(i) } as usize;
            let pi = unsafe { *atoms.pos.get_unchecked(i) };

            if has_self {
                // Start after atom i's position in the sorted array — all subsequent atoms
                // in this bin have j > i (locals in order, ghosts have index >= nlocal > i).
                let self_start = unsafe { *atom_sorted_idx.get_unchecked(i) } as usize + 1;
                let end = unsafe { *bin_start.get_unchecked(my_cell + 1) } as usize;
                for m in self_start..end {
                    let pj = unsafe { *sorted_pos.get_unchecked(m) };
                    let dx = pj[0] - pi[0];
                    let dy = pj[1] - pi[1];
                    let dz = pj[2] - pi[2];
                    let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                    if r2 < cutoff_sq {
                        let j = unsafe { *sorted_atoms.get_unchecked(m) };
                        push_index!(j);
                    }
                }
            }

            for &offset in &stencil_forward {
                let c = (my_cell as i32 + offset) as usize;
                let start = unsafe { *bin_start.get_unchecked(c) } as usize;
                let end = unsafe { *bin_start.get_unchecked(c + 1) } as usize;
                for m in start..end {
                    let pj = unsafe { *sorted_pos.get_unchecked(m) };
                    let dx = pj[0] - pi[0];
                    let dy = pj[1] - pi[1];
                    let dz = pj[2] - pi[2];
                    let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                    if r2 < cutoff_sq {
                        let j = unsafe { *sorted_atoms.get_unchecked(m) };
                        push_index!(j);
                    }
                }
            }
        }
    } else {
        // Slow path: per-pair skin computation
        let skin_fraction_sq = skin_fraction * skin_fraction;
        for i in 0..nlocal {
            offsets.push(nidx as u32);
            let my_cell = unsafe { *atom_cell.get_unchecked(i) } as usize;
            let pi = unsafe { *atoms.pos.get_unchecked(i) };
            let si = unsafe { *atoms.cutoff_radius.get_unchecked(i) };

            if has_self {
                let self_start = unsafe { *atom_sorted_idx.get_unchecked(i) } as usize + 1;
                let end = unsafe { *bin_start.get_unchecked(my_cell + 1) } as usize;
                for m in self_start..end {
                    let j = unsafe { *sorted_atoms.get_unchecked(m) } as usize;
                    let pj = unsafe { *sorted_pos.get_unchecked(m) };
                    let dx = pj[0] - pi[0];
                    let dy = pj[1] - pi[1];
                    let dz = pj[2] - pi[2];
                    let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                    let sum_skin = si + unsafe { *atoms.cutoff_radius.get_unchecked(j) };
                    if r2 < sum_skin * sum_skin * skin_fraction_sq {
                        push_index!(j as u32);
                    }
                }
            }

            for &offset in &stencil_forward {
                let c = (my_cell as i32 + offset) as usize;
                let start = unsafe { *bin_start.get_unchecked(c) } as usize;
                let end = unsafe { *bin_start.get_unchecked(c + 1) } as usize;
                for m in start..end {
                    let j = unsafe { *sorted_atoms.get_unchecked(m) } as usize;
                    let pj = unsafe { *sorted_pos.get_unchecked(m) };
                    let dx = pj[0] - pi[0];
                    let dy = pj[1] - pi[1];
                    let dz = pj[2] - pi[2];
                    let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                    let sum_skin = si + unsafe { *atoms.cutoff_radius.get_unchecked(j) };
                    if r2 < sum_skin * sum_skin * skin_fraction_sq {
                        push_index!(j as u32);
                    }
                }
            }
        }
    }
    // SAFETY: nidx elements were written via raw pointer; all < capacity.
    unsafe { indices.set_len(nidx) };
    offsets.push(nidx as u32);

    neighbor.neighbor_offsets = offsets;
    neighbor.neighbor_indices = indices;
    neighbor.bin_stencil_forward = stencil_forward;
    neighbor.bin_atom_cell = atom_cell;
    neighbor.bin_count_arr = bin_count_arr;
    neighbor.bin_start = bin_start;
    neighbor.bin_sorted_atoms = sorted_atoms;
    neighbor.bin_atom_sorted_idx = atom_sorted_idx;
    neighbor.bin_sorted_pos = sorted_pos;
}

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::Atom;
    
    fn push_atom(atom: &mut Atom, tag: u32, pos: [f64; 3], radius: f64) {
        atom.push_test_atom(tag, pos, radius, 1.0);
    }

    #[test]
    fn brute_force_finds_close_pair() {
        let mut app = App::new();
        let mut atom = Atom::new();
        push_atom(&mut atom, 0, [0.0, 0.0, 0.0], 0.5);
        push_atom(&mut atom, 1, [0.5, 0.0, 0.0], 0.5);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = 1.0;

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_update_system(brute_force_neighbor_list, ScheduleSet::Neighbor);
        app.organize_systems();
        app.run();

        let n = app.get_resource_ref::<Neighbor>().unwrap();
        assert_eq!(n.neighbor_list.len(), 1);
        let (i, j) = n.neighbor_list[0];
        assert!(i == 0 && j == 1);
    }

    #[test]
    fn brute_force_misses_distant_pair() {
        let mut app = App::new();
        let mut atom = Atom::new();
        push_atom(&mut atom, 0, [0.0, 0.0, 0.0], 0.1);
        push_atom(&mut atom, 1, [5.0, 0.0, 0.0], 0.1);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = 1.0;

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_update_system(brute_force_neighbor_list, ScheduleSet::Neighbor);
        app.organize_systems();
        app.run();

        let n = app.get_resource_ref::<Neighbor>().unwrap();
        assert_eq!(n.neighbor_list.len(), 0);
    }

    #[test]
    fn bin_neighbor_list_finds_close_pair() {
        let mut app = App::new();
        let mut atom = Atom::new();
        push_atom(&mut atom, 0, [0.5, 0.5, 0.5], 0.5);
        push_atom(&mut atom, 1, [1.0, 0.5, 0.5], 0.5);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut domain = mddem_core::Domain::new();
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [2.0, 2.0, 2.0];
        domain.sub_length = [2.0, 2.0, 2.0];

        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = 1.0;
        neighbor.bin_min_size = 1.0;

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(domain);
        app.add_resource(NeighborConfig::default());
        app.add_resource(mddem_core::CommResource(Box::new(
            mddem_core::SingleProcessComm::new(),
        )));
        app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
        app.add_update_system(bin_neighbor_list, ScheduleSet::Neighbor);
        app.organize_systems();
        app.setup();
        app.run();

        let n = app.get_resource_ref::<Neighbor>().unwrap();
        assert!(
            n.neighbor_indices.len() >= 1,
            "bin neighbor list should find the close pair"
        );
        // Check CSR: atom 0's neighbors should contain 1
        let start = n.neighbor_offsets[0] as usize;
        let end = n.neighbor_offsets[1] as usize;
        let has_pair = n.neighbor_indices[start..end].contains(&1u32);
        assert!(has_pair, "pair (0,1) should be in CSR neighbors");
    }

    #[test]
    fn pair_iter_matches_manual() {
        let mut neighbor = Neighbor::new();
        // 3 local atoms: atom 0 -> [1, 2], atom 1 -> [2], atom 2 -> []
        neighbor.neighbor_offsets = vec![0, 2, 3, 3];
        neighbor.neighbor_indices = vec![1, 2, 2];

        let pairs: Vec<(usize, usize)> = neighbor.pairs(3).collect();
        assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn pair_iter_empty() {
        let mut neighbor = Neighbor::new();
        neighbor.neighbor_offsets = vec![0, 0, 0];
        neighbor.neighbor_indices = vec![];

        let pairs: Vec<(usize, usize)> = neighbor.pairs(2).collect();
        assert!(pairs.is_empty());

        // Also test zero local atoms
        let pairs2: Vec<(usize, usize)> = neighbor.pairs(0).collect();
        assert!(pairs2.is_empty());
    }

    #[test]
    fn sweep_and_prune_finds_close_pair() {
        let mut app = App::new();
        let mut atom = Atom::new();
        push_atom(&mut atom, 0, [0.0, 0.0, 0.0], 0.5);
        push_atom(&mut atom, 1, [0.5, 0.0, 0.0], 0.5);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = 1.0;

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(mddem_core::Domain::new());
        app.add_resource(mddem_core::CommResource(Box::new(
            mddem_core::SingleProcessComm::new(),
        )));
        app.add_update_system(sweep_and_prune_neighbor_list, ScheduleSet::Neighbor);
        app.organize_systems();
        app.run();

        let n = app.get_resource_ref::<Neighbor>().unwrap();
        assert_eq!(n.neighbor_list.len(), 1);
    }

    // ── Brute-force vs bin-based neighbor list comparison ──────────────────

    /// Reference O(N²) all-pairs neighbor finder that doesn't use ghost atoms
    /// or any accelerated algorithm. Pure distance-based cutoff check.
    fn reference_all_pairs(
        positions: &[[f64; 3]],
        cutoff_radii: &[f64],
        skin_fraction: f64,
        nlocal: usize,
    ) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..nlocal {
            for j in (i + 1)..positions.len() {
                let dx = positions[j][0] - positions[i][0];
                let dy = positions[j][1] - positions[i][1];
                let dz = positions[j][2] - positions[i][2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let cutoff = (cutoff_radii[i] + cutoff_radii[j]) * skin_fraction;
                if dist < cutoff {
                    pairs.push((i, j));
                }
            }
        }
        pairs.sort();
        pairs
    }

    #[test]
    fn brute_force_matches_reference_all_pairs() {
        // Create a system of 20 particles at pseudo-random positions
        // and verify brute_force_neighbor_list matches our reference.
        let n = 20;
        let skin_fraction = 1.0;
        let radius = 0.5;

        let mut atom = Atom::new();
        // Place atoms in a pseudo-random but deterministic pattern
        for i in 0..n {
            let x = (i as f64 * 0.37).sin() * 2.0 + 3.0;
            let y = (i as f64 * 0.73).cos() * 2.0 + 3.0;
            let z = (i as f64 * 1.13).sin() * 2.0 + 3.0;
            push_atom(&mut atom, i as u32, [x, y, z], radius);
        }
        atom.nlocal = n as u32;
        atom.natoms = n as u64;

        // Reference calculation
        let ref_pairs = reference_all_pairs(
            &atom.pos[..n],
            &atom.cutoff_radius[..n],
            skin_fraction,
            n,
        );

        // Build via brute_force_neighbor_list
        let mut app = App::new();
        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = skin_fraction;
        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_update_system(brute_force_neighbor_list, ScheduleSet::Neighbor);
        app.organize_systems();
        app.run();

        let neigh = app.get_resource_ref::<Neighbor>().unwrap();
        let mut bf_pairs: Vec<(usize, usize)> = neigh.neighbor_list.clone();
        // Normalize: ensure (min, max) ordering
        for p in bf_pairs.iter_mut() {
            if p.0 > p.1 {
                *p = (p.1, p.0);
            }
        }
        bf_pairs.sort();
        bf_pairs.dedup();

        assert_eq!(
            bf_pairs, ref_pairs,
            "Brute-force neighbor list doesn't match reference.\nBF: {:?}\nRef: {:?}",
            bf_pairs, ref_pairs
        );
    }

    #[test]
    fn bin_neighbor_list_matches_brute_force() {
        // Create a system and verify that bin-based neighbor list finds
        // exactly the same pairs as brute-force.
        let n = 30;
        let skin_fraction = 1.0;
        let radius = 0.3;

        let mut positions = Vec::new();
        for i in 0..n {
            let x = (i as f64 * 0.31).sin() * 1.5 + 2.0;
            let y = (i as f64 * 0.67).cos() * 1.5 + 2.0;
            let z = (i as f64 * 0.97).sin() * 1.5 + 2.0;
            positions.push([x, y, z]);
        }

        // Reference all-pairs
        let cutoffs: Vec<f64> = vec![radius; n];
        let ref_pairs = reference_all_pairs(&positions, &cutoffs, skin_fraction, n);

        // Brute-force neighbor list
        let mut app_bf = App::new();
        let mut atom_bf = Atom::new();
        for i in 0..n {
            push_atom(&mut atom_bf, i as u32, positions[i], radius);
        }
        atom_bf.nlocal = n as u32;
        atom_bf.natoms = n as u64;
        let mut neighbor_bf = Neighbor::new();
        neighbor_bf.skin_fraction = skin_fraction;
        app_bf.add_resource(atom_bf);
        app_bf.add_resource(neighbor_bf);
        app_bf.add_update_system(brute_force_neighbor_list, ScheduleSet::Neighbor);
        app_bf.organize_systems();
        app_bf.run();

        let neigh_bf = app_bf.get_resource_ref::<Neighbor>().unwrap();
        let mut bf_pairs: Vec<(usize, usize)> = neigh_bf.neighbor_list.clone();
        for p in bf_pairs.iter_mut() {
            if p.0 > p.1 { *p = (p.1, p.0); }
        }
        bf_pairs.sort();
        bf_pairs.dedup();

        // Bin-based neighbor list
        let mut app_bin = App::new();
        let mut atom_bin = Atom::new();
        for i in 0..n {
            push_atom(&mut atom_bin, i as u32, positions[i], radius);
        }
        atom_bin.nlocal = n as u32;
        atom_bin.natoms = n as u64;

        let mut domain = mddem_core::Domain::new();
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [4.0, 4.0, 4.0];
        domain.sub_length = [4.0, 4.0, 4.0];

        let mut neighbor_bin = Neighbor::new();
        neighbor_bin.skin_fraction = skin_fraction;
        neighbor_bin.bin_min_size = 0.5;

        app_bin.add_resource(atom_bin);
        app_bin.add_resource(neighbor_bin);
        app_bin.add_resource(domain);
        app_bin.add_resource(NeighborConfig::default());
        app_bin.add_resource(mddem_core::CommResource(Box::new(
            mddem_core::SingleProcessComm::new(),
        )));
        app_bin.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
        app_bin.add_update_system(bin_neighbor_list, ScheduleSet::Neighbor);
        app_bin.organize_systems();
        app_bin.setup();
        app_bin.run();

        let neigh_bin = app_bin.get_resource_ref::<Neighbor>().unwrap();
        // Extract pairs from CSR format
        let nlocal = n;
        let mut bin_pairs = Vec::new();
        for i in 0..nlocal {
            if i + 1 >= neigh_bin.neighbor_offsets.len() {
                break;
            }
            let start = neigh_bin.neighbor_offsets[i] as usize;
            let end = neigh_bin.neighbor_offsets[i + 1] as usize;
            for k in start..end {
                let j = neigh_bin.neighbor_indices[k] as usize;
                let pair = if i < j { (i, j) } else { (j, i) };
                bin_pairs.push(pair);
            }
        }
        bin_pairs.sort();
        bin_pairs.dedup();

        // Verify: bin-based should find at least all pairs that brute-force finds
        // (bin-based might find more if there are edge effects, but shouldn't miss any)
        for pair in &ref_pairs {
            assert!(
                bin_pairs.contains(pair),
                "Bin neighbor list missed pair {:?} found by reference",
                pair
            );
        }

        // Also verify brute-force matches reference
        assert_eq!(
            bf_pairs, ref_pairs,
            "Brute-force doesn't match reference for N=30 system"
        );
    }

    #[test]
    fn no_spurious_pairs_in_brute_force() {
        // Verify that brute-force doesn't include pairs beyond the cutoff.
        // Place two atoms far apart (distance 5.0) with small radii (0.1).
        // Cutoff = (0.1 + 0.1) * 1.0 = 0.2. Distance 5.0 >> 0.2.
        let mut atom = Atom::new();
        push_atom(&mut atom, 0, [0.0, 0.0, 0.0], 0.1);
        push_atom(&mut atom, 1, [5.0, 0.0, 0.0], 0.1);
        push_atom(&mut atom, 2, [0.15, 0.0, 0.0], 0.1); // close to atom 0
        atom.nlocal = 3;
        atom.natoms = 3;

        let ref_pairs = reference_all_pairs(
            &atom.pos[..3],
            &atom.cutoff_radius[..3],
            1.0,
            3,
        );

        let mut app = App::new();
        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = 1.0;
        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_update_system(brute_force_neighbor_list, ScheduleSet::Neighbor);
        app.organize_systems();
        app.run();

        let neigh = app.get_resource_ref::<Neighbor>().unwrap();
        let mut bf_pairs: Vec<(usize, usize)> = neigh.neighbor_list.clone();
        for p in bf_pairs.iter_mut() {
            if p.0 > p.1 { *p = (p.1, p.0); }
        }
        bf_pairs.sort();
        bf_pairs.dedup();

        // Should only find (0, 2), not (0, 1) or (1, 2)
        assert_eq!(
            bf_pairs, ref_pairs,
            "Spurious pairs detected.\nBF: {:?}\nRef: {:?}",
            bf_pairs, ref_pairs
        );
        assert_eq!(ref_pairs.len(), 1, "Should find exactly 1 pair: (0,2)");
        assert_eq!(ref_pairs[0], (0, 2));
    }

    #[test]
    fn varied_cutoff_radii_neighbor_list() {
        // Test with different cutoff radii per atom.
        // Atom 0: radius=0.5 at origin
        // Atom 1: radius=0.1 at (0.5, 0, 0) -- cutoff = 0.6 > 0.5 ✓ neighbor
        // Atom 2: radius=0.1 at (0.7, 0, 0) -- cutoff = 0.6 < 0.7 ✗ not neighbor
        let mut atom = Atom::new();
        push_atom(&mut atom, 0, [0.0, 0.0, 0.0], 0.5);
        push_atom(&mut atom, 1, [0.5, 0.0, 0.0], 0.1);
        push_atom(&mut atom, 2, [0.7, 0.0, 0.0], 0.1);
        atom.nlocal = 3;
        atom.natoms = 3;

        let ref_pairs = reference_all_pairs(
            &atom.pos[..3],
            &atom.cutoff_radius[..3],
            1.0,
            3,
        );

        let mut app = App::new();
        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = 1.0;
        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_update_system(brute_force_neighbor_list, ScheduleSet::Neighbor);
        app.organize_systems();
        app.run();

        let neigh = app.get_resource_ref::<Neighbor>().unwrap();
        let mut bf_pairs: Vec<(usize, usize)> = neigh.neighbor_list.clone();
        for p in bf_pairs.iter_mut() {
            if p.0 > p.1 { *p = (p.1, p.0); }
        }
        bf_pairs.sort();
        bf_pairs.dedup();

        assert_eq!(bf_pairs, ref_pairs);
        // (0,1): dist=0.5, cutoff=0.6 ✓
        // (0,2): dist=0.7, cutoff=0.6 ✗
        // (1,2): dist=0.2, cutoff=0.2 ✗ (exactly at boundary, not less than)
        assert!(
            ref_pairs.contains(&(0, 1)),
            "Should find pair (0,1)"
        );
    }
}

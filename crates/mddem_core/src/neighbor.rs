//! Neighbor list construction for particle simulations.
//!
//! Uses spatial binning with a precomputed stencil of neighbor cells for O(N)
//! expected neighbor finding. Produces a **half neighbor list** in CSR
//! (Compressed Sparse Row) format: each local atom `i` stores its neighbor
//! indices `j > i` (or ghost atoms) in a flat array, with offsets delimiting
//! each atom's neighbors. Use [`Neighbor::pairs()`] to iterate over `(i, j)`
//! pairs efficiently.
//!
//! # Rebuild strategies
//! 
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

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{Atom, AtomDataRegistry, CommResource, CommState, Config, Domain, ParticleSimScheduleSet, ScheduleSetupSet};

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
    /// Newton's third law optimization: true = half neighbor list + reverse comm,
    /// false = full neighbor list, no reverse comm.
    #[serde(default = "default_true")]
    pub newton: bool,
}

impl Default for NeighborConfig {
    fn default() -> Self {
        NeighborConfig {
            skin_fraction: 1.0,
            bin_size: 1.0,
            every: 0,
            check: true,
            sort_every: 1000,
            newton: true,
        }
    }
}

/// Neighbor list state: CSR indices, bin grid, and rebuild tracking.
///
/// The primary output is the CSR neighbor list stored in [`neighbor_offsets`](Self::neighbor_offsets)
/// and [`neighbor_indices`](Self::neighbor_indices). Use [`pairs()`](Self::pairs) to iterate
/// over `(i, j)` neighbor pairs.
pub struct Neighbor {
    /// Multiplier on pairwise cutoff: pair cutoff = `(r_i + r_j) * skin_fraction`.
    /// Values > 1.0 add a "skin" buffer to reduce rebuild frequency.
    pub skin_fraction: f64,
    /// CSR row offsets: `neighbor_offsets[i]..neighbor_offsets[i+1]` gives the range
    /// of neighbor indices for local atom `i`. Length = `nlocal + 1`.
    pub neighbor_offsets: Vec<u32>,
    /// CSR column indices: flat array of neighbor atom indices (local or ghost).
    pub neighbor_indices: Vec<u32>,
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
    /// Newton's third law optimization: `true` = half neighbor list + reverse comm,
    /// `false` = full neighbor list, skip reverse comm.
    pub newton: bool,
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
            neighbor_offsets: Vec::new(),
            neighbor_indices: Vec::new(),
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
            newton: true,
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

    // Ghost layers per side: enough bins to cover the ghost zone + stencil reach.
    // When ghost_cutoff > pair cutoff (e.g., dem_clump extends it for sub-sphere
    // offsets), local atoms can end up in ghost bins. The stencil from those bins
    // must not exceed the grid, so we add stencil range on top of ghost layers.
    let ghost_cut = if domain.ghost_cutoff > max_cutoff { domain.ghost_cutoff } else { max_cutoff };
    let stencil_range_x = (max_cutoff / actual_bin_size[0]).ceil() as i32;
    let stencil_range_y = (max_cutoff / actual_bin_size[1]).ceil() as i32;
    let stencil_range_z = (max_cutoff / actual_bin_size[2]).ceil() as i32;
    let sx = (ghost_cut / actual_bin_size[0]).ceil() as i32 + stencil_range_x;
    let sy = (ghost_cut / actual_bin_size[1]).ceil() as i32 + stencil_range_y;
    let sz = (ghost_cut / actual_bin_size[2]).ceil() as i32 + stencil_range_z;

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
/// Uses spatial binning with CSR layout for O(N) expected neighbor finding.
///
/// # Systems registered
///
/// - **Setup**: [`neighbor_read_input`] (reads `[neighbor]` config) and
///   [`neighbor_setup`] (computes bin grid, ghost cutoff).
/// - **Update**: [`decide_rebuild`] (displacement check),
///   [`sort_atoms_by_bin`] (cache-locality reordering),
///   [`bin_neighbor_list`] (bin-based neighbor construction).
pub struct NeighborPlugin;

impl Plugin for NeighborPlugin {
    fn provides(&self) -> Vec<&str> {
        vec!["neighbor_list"]
    }

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
sort_every = 1000
# Newton's third law optimization (true = half list + reverse comm, false = full list)
newton = true"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<NeighborConfig>(app, "neighbor");

        app.add_resource(Neighbor::new())
            .add_resource(CurrentState(CommState::FullRebuild))
            .add_setup_system(neighbor_read_input, ScheduleSetupSet::Setup)
            .add_setup_system(neighbor_setup.label("neighbor_setup"), ScheduleSetupSet::PostSetup);
        
        app.add_update_system(
                decide_rebuild
                    .label("decide_rebuild")
                    .before(crate::remove_ghost_atoms)
                    .run_if(in_state(CommState::CommunicateOnly)),
                ParticleSimScheduleSet::PostInitialIntegration,
            );
        app.add_update_system(
            sort_atoms_by_bin
                .label("sort_atoms")
                .before(crate::borders),
            ParticleSimScheduleSet::PreNeighbor,
        );
        app.add_update_system(
            bin_neighbor_list.run_if(in_state(CommState::FullRebuild)),
            ParticleSimScheduleSet::Neighbor,
        );
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
    neighbor.newton = config.newton;
    if comm.rank() == 0 {
        let rebuild_str = if config.every == 0 {
            "displacement".to_string()
        } else if config.check {
            format!("every {} + check", config.every)
        } else {
            format!("every {}", config.every)
        };
        println!(
            "Neighbor: skin_fraction={} bin_size={} rebuild={} newton={}",
            config.skin_fraction, config.bin_size, rebuild_str, if config.newton { "on" } else { "off" }
        );
    }
}

/// Setup system: computes bin grid, ghost cutoff, and stencil from atom cutoff radii.
///
/// Runs at [`ScheduleSetupSet::PostSetup`] after atoms are created. Determines:
/// - `max_cutoff = 2 * max_skin * skin_fraction` (largest pairwise neighbor distance)
/// - `ghost_cutoff = max_cutoff + displacement_buffer` (communication distance for ghosts)
/// - Bin grid dimensions, stencil offsets, and PBC flags
pub fn neighbor_setup(_config: Res<NeighborConfig>, mut neighbor: ResMut<Neighbor>, mut domain: ResMut<Domain>, atoms: ResMut<Atom>, comm: Res<CommResource>) {
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
    let ghost_cut = (max_cutoff + 2.0 * displacement_buffer).max(domain.ghost_cutoff);
    neighbor.ghost_cutoff = ghost_cut;
    neighbor.cached_max_cutoff = max_cutoff;
    domain.ghost_cutoff = ghost_cut;
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
    neighbor.pbc_flags = domain.periodic_flags();

    if comm.rank() == 0 {
        println!(
            "Neighbor: bins {}x{}x{} (with ghost layers), {} forward stencil cells",
            neighbor.bin_count[0], neighbor.bin_count[1], neighbor.bin_count[2],
            neighbor.bin_stencil_forward.len()
        );
    }
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
fn needs_rebuild(atoms: &Atom, neighbor: &Neighbor, comm_state: &CurrentState<CommState>) -> bool {
    let nlocal = atoms.nlocal as usize;
    // Always rebuild on first call or local atom count change
    if neighbor.last_build_pos.len() != nlocal || nlocal == 0 {
        return true;
    }
    // If state is FullRebuild, a full rebuild was requested (first step,
    // or decide_rebuild detected displacement exceeded threshold).
    if comm_state.0 == CommState::FullRebuild {
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
/// Gated by `run_if(in_state(CommState::CommunicateOnly))` — only checks when
/// the neighbor list is still considered valid.
///
/// Checks displacement to decide if a full rebuild is needed this step.
/// If so, sets CommState to FullRebuild so that remove_ghost_atoms / exchange /
/// full borders all run.
///
/// Uses all_reduce to ensure ALL ranks agree — if any rank needs a rebuild,
/// all ranks do the full rebuild (required for MPI send/recv pattern matching).
pub fn decide_rebuild(
    atoms: Res<Atom>,
    mut neighbor: ResMut<Neighbor>,
    comm: Res<CommResource>,
    domain: Res<Domain>,
    mut comm_state: ResMut<CurrentState<CommState>>,
) {
    let mut local_needs = if needs_rebuild(&atoms, &neighbor, &comm_state) { 1.0 } else { 0.0 };
    // Detect atoms about to cross a periodic boundary. After initial_integration
    // moved atoms but before pbc() wraps them, atoms outside the global box are
    // exactly the ones that will wrap. PBC wrap requires full rebuild so exchange
    // can migrate them to the correct sub-domain.
    if local_needs == 0.0 {
        let low = domain.boundaries_low;
        let high = domain.boundaries_high;
        let periodic = domain.periodic_flags();
        for i in 0..atoms.nlocal as usize {
            for d in 0..3 {
                if periodic[d] && (atoms.pos[i][d] < low[d] || atoms.pos[i][d] >= high[d]) {
                    local_needs = 1.0;
                    break;
                }
            }
            if local_needs > 0.0 {
                break;
            }
        }
    }
    // Any rank needing rebuild forces all ranks to rebuild
    let global_needs = comm.all_reduce_sum_f64(local_needs);
    if global_needs > 0.0 {
        comm_state.0 = CommState::FullRebuild;
    } else {
        neighbor.steps_since_build += 1;
    }
}

/// Reorder local atoms by spatial bin for improved cache locality.
///
/// Runs at [`ParticleSimScheduleSet::PreNeighbor`] (before ghost communication). Atoms are
/// sorted by their bin cell index so that spatially nearby atoms are contiguous
/// in memory, improving cache hit rates during neighbor list construction and
/// force computation.
///
/// Sorting is triggered either periodically (every `sort_every` steps) or when
/// a neighbor rebuild is needed. The permutation is applied to all atom arrays
/// (via [`AtomDataRegistry`]) and ghost `origin_index` values are updated.
pub fn sort_atoms_by_bin(mut atoms: ResMut<Atom>, mut neighbor: ResMut<Neighbor>, comm: Res<CommResource>, registry: Res<AtomDataRegistry>, mut domain: ResMut<Domain>, mut comm_state: ResMut<CurrentState<CommState>>) {
    // Recompute bin grid if domain bounds changed (e.g., shrink-wrap).
    // Must happen before sorting since sorting depends on bin parameters.
    if domain.bounds_changed {
        recompute_bins(&mut neighbor, &domain, comm.size());
        domain.bounds_changed = false;
        comm_state.0 = CommState::FullRebuild;
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
                comm_state.0 = CommState::FullRebuild;
            }
        }
        return;
    }

    let rebuild_needed = needs_rebuild(&atoms, &neighbor, &comm_state);
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
            comm_state.0 = CommState::FullRebuild;
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
    atoms: Res<Atom>,
    mut neighbor: ResMut<Neighbor>,
    mut comm_state: ResMut<CurrentState<CommState>>,
) {
    let nlocal = atoms.nlocal as usize;
    let total = atoms.len();

    comm_state.0 = CommState::CommunicateOnly;
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
    let newton = neighbor.newton;
    // Take stencil to avoid borrow conflict with neighbor_indices
    let stencil_forward = std::mem::take(&mut neighbor.bin_stencil_forward);
    let stencil_full = std::mem::take(&mut neighbor.bin_stencil);

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
        // Select stencil based on newton flag
        let stencil = if newton { &stencil_forward } else { &stencil_full };
        for i in 0..nlocal {
            offsets.push(nidx as u32);
            let my_cell = unsafe { *atom_cell.get_unchecked(i) } as usize;
            let pi = unsafe { *atoms.pos.get_unchecked(i) };

            if has_self {
                let cell_start = unsafe { *bin_start.get_unchecked(my_cell) } as usize;
                let end = unsafe { *bin_start.get_unchecked(my_cell + 1) } as usize;
                if newton {
                    // Half list: start after atom i's position — all subsequent atoms
                    // in this bin have j > i (locals in order, ghosts have index >= nlocal > i).
                    let self_start = unsafe { *atom_sorted_idx.get_unchecked(i) } as usize + 1;
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
                } else {
                    // Full list: scan all atoms in self-cell except i itself
                    for m in cell_start..end {
                        let j = unsafe { *sorted_atoms.get_unchecked(m) };
                        if j as usize == i { continue; }
                        let pj = unsafe { *sorted_pos.get_unchecked(m) };
                        let dx = pj[0] - pi[0];
                        let dy = pj[1] - pi[1];
                        let dz = pj[2] - pi[2];
                        let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                        if r2 < cutoff_sq {
                            push_index!(j);
                        }
                    }
                }
            }

            for &offset in stencil.iter() {
                // newton=true: stencil_forward has only offset > 0 (skip self cell)
                // newton=false: stencil_full includes offset <= 0 but self cell (0) handled above
                if !newton && offset == 0 { continue; }
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
        let stencil = if newton { &stencil_forward } else { &stencil_full };
        for i in 0..nlocal {
            offsets.push(nidx as u32);
            let my_cell = unsafe { *atom_cell.get_unchecked(i) } as usize;
            let pi = unsafe { *atoms.pos.get_unchecked(i) };
            let si = unsafe { *atoms.cutoff_radius.get_unchecked(i) };

            if has_self {
                let cell_start = unsafe { *bin_start.get_unchecked(my_cell) } as usize;
                let end = unsafe { *bin_start.get_unchecked(my_cell + 1) } as usize;
                if newton {
                    let self_start = unsafe { *atom_sorted_idx.get_unchecked(i) } as usize + 1;
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
                } else {
                    for m in cell_start..end {
                        let j = unsafe { *sorted_atoms.get_unchecked(m) } as usize;
                        if j == i { continue; }
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

            for &offset in stencil.iter() {
                if !newton && offset == 0 { continue; }
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
    neighbor.bin_stencil = stencil_full;
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
    use crate::Atom;
    
    fn push_atom(atom: &mut Atom, tag: u32, pos: [f64; 3], radius: f64) {
        atom.push_test_atom(tag, pos, radius, 1.0);
    }

    #[test]
    fn bin_neighbor_list_finds_close_pair() {
        let mut app = App::new();
        let mut atom = Atom::new();
        push_atom(&mut atom, 0, [0.5, 0.5, 0.5], 0.5);
        push_atom(&mut atom, 1, [1.0, 0.5, 0.5], 0.5);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut domain = crate::Domain::new();
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
        app.add_resource(CurrentState(CommState::FullRebuild));
        app.add_resource(crate::CommResource(Box::new(
            crate::SingleProcessComm::new(),
        )));
        app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
        app.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
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

    // ── Reference vs bin-based neighbor list comparison ──────────────────

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
    fn bin_neighbor_list_matches_reference() {
        // Create a system and verify that bin-based neighbor list finds
        // exactly the same pairs as the reference all-pairs calculation.
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

        // Bin-based neighbor list
        let mut app_bin = App::new();
        let mut atom_bin = Atom::new();
        for i in 0..n {
            push_atom(&mut atom_bin, i as u32, positions[i], radius);
        }
        atom_bin.nlocal = n as u32;
        atom_bin.natoms = n as u64;

        let mut domain = crate::Domain::new();
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
        app_bin.add_resource(CurrentState(CommState::FullRebuild));
        app_bin.add_resource(crate::CommResource(Box::new(
            crate::SingleProcessComm::new(),
        )));
        app_bin.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
        app_bin.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
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

        // Verify: bin-based should find all pairs that reference finds
        for pair in &ref_pairs {
            assert!(
                bin_pairs.contains(pair),
                "Bin neighbor list missed pair {:?} found by reference",
                pair
            );
        }
    }
}

//! Neighbor list construction: brute-force, sweep-and-prune, and CSR bin-based algorithms.
//!
//! Provides displacement-based and periodic rebuild strategies with configurable skin fraction.

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use mddem_core::{Atom, AtomDataRegistry, CommResource, Config, Domain};

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
}

impl Default for NeighborConfig {
    fn default() -> Self {
        NeighborConfig {
            skin_fraction: 1.0,
            bin_size: 1.0,
            every: 0,
            check: true,
            sort_every: 1000,
        }
    }
}

/// Algorithm used to build the neighbor list.
pub enum NeighborStyle {
    BruteForce,
    SweepAndPrune,
    Bin,
}

/// Neighbor list state: pair lists, CSR indices, bin grid, and rebuild tracking.
pub struct Neighbor {
    pub skin_fraction: f64,
    pub neighbor_list: Vec<(usize, usize)>,
    // Merged CSR neighbor list: neighbor_offsets[i]..neighbor_offsets[i+1] gives j indices for atom i
    pub neighbor_offsets: Vec<u32>,    // length = nlocal + 1
    pub neighbor_indices: Vec<u32>,   // flat j indices (all pairs, local + ghost)
    pub sweep_and_prune: Vec<(usize, f64)>,
    pub bin_min_size: f64,
    pub bin_size: Vector3<f64>,
    pub bin_count: Vector3<i32>,
    pub last_build_pos: Vec<[f64; 3]>,
    pub steps_since_build: usize,
    pub last_build_total: usize,
    pub pbc_box: [f64; 3],           // box dimensions for minimum-image displacement check
    pub pbc_flags: [bool; 3],        // which axes are periodic
    pub bin_stencil: Vec<i32>,    // flat offsets for neighbor cell stencil (full)
    pub bin_stencil_forward: Vec<i32>, // offsets > 0 (half stencil for local-local dedup)
    pub bin_stencil_self: bool,        // whether self-cell (offset 0) is in stencil
    pub bin_origin: Vector3<f64>, // lower-left corner of bin grid (sub_domain_low - bin_size)
    pub bin_total_cells: usize,   // nx * ny * nz (including ghost layers)
    pub every: usize,             // rebuild every N steps (0 = displacement only)
    pub check: bool,              // also check displacement when every > 0
    pub ghost_cutoff: f64,        // max distance for ghost atom communication
    pub cached_min_skin: f64,     // cached minimum skin value, computed at rebuild time
    pub sort_every: usize,        // sort atoms by bin every N steps (0 = disabled)
    pub sort_counter: usize,      // steps since last sort
    // Persistent bin arrays (reused across rebuilds to avoid per-rebuild allocations)
    pub bin_atom_cell: Vec<u32>,
    pub bin_count_arr: Vec<u32>,
    pub bin_start: Vec<u32>,
    pub bin_sorted_atoms: Vec<u32>,
    // Per-atom position in sorted_atoms array (for self-cell skip optimization)
    pub bin_atom_sorted_idx: Vec<u32>,
    // Sorted position cache for cache-friendly inner loop access
    pub bin_sorted_pos: Vec<[f64; 3]>,
    // Cached uniform cutoff squared (Some when all atoms have the same skin)
    pub cached_uniform_cutoff_sq: Option<f64>,
}

impl Default for Neighbor {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over `(i, j)` neighbor pairs from the CSR neighbor list.
pub struct PairIter<'a> {
    offsets: &'a [u32],
    indices: &'a [u32],
    nlocal: usize,
    i: usize,
    k: usize,
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

    pub fn new() -> Self {
        Neighbor {
            skin_fraction: 1.0,
            neighbor_list: Vec::new(),
            neighbor_offsets: Vec::new(),
            neighbor_indices: Vec::new(),
            sweep_and_prune: Vec::new(),
            bin_min_size: 1.0,
            bin_size: Vector3::new(1.0, 1.0, 1.0),
            bin_count: Vector3::new(1, 1, 1),
            last_build_pos: Vec::new(),
            steps_since_build: 0,
            last_build_total: 0,
            pbc_box: [0.0; 3],
            pbc_flags: [false; 3],
            bin_stencil: Vec::new(),
            bin_stencil_forward: Vec::new(),
            bin_stencil_self: false,
            bin_origin: Vector3::new(0.0, 0.0, 0.0),
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
        }
    }
}

/// Registers neighbor list construction and rebuild systems.
pub struct NeighborPlugin {
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
                decide_rebuild.label("decide_rebuild").before("remove_ghost_atoms"),
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
                        .before("borders"),
                    ScheduleSet::PreNeighbor,
                );
                app.add_update_system(bin_neighbor_list, ScheduleSet::Neighbor);
            }
        }
    }
}

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

pub fn neighbor_setup(mut neighbor: ResMut<Neighbor>, mut domain: ResMut<Domain>, atoms: Res<Atom>, comm: Res<CommResource>) {
    // Compute max neighbor cutoff = (skin_i + skin_j) * skin_fraction = 2 * max_skin * skin_fraction
    // Use global reduction: at PostSetup, atoms may only be on rank 0 (before exchange).
    let local_max_skin = atoms.skin.iter().cloned().fold(0.0f64, f64::max);
    let max_skin = -comm.all_reduce_min_f64(-local_max_skin); // global max via negated min
    let max_cutoff = 2.0 * max_skin * neighbor.skin_fraction;
    // Add displacement buffer to ghost_cutoff so atoms don't drift in/out of the
    // ghost zone between neighbor rebuilds. Without this padding, ghost count
    // fluctuates every step, forcing unnecessary neighbor rebuilds.
    // Max per-atom displacement before rebuild = (skin_fraction - 1) * min_skin.
    // Two atoms can each move this far, so buffer = 2 * displacement.
    let local_min_skin = atoms.skin.iter().cloned().fold(f64::MAX, f64::min);
    let min_skin = comm.all_reduce_min_f64(local_min_skin);
    let displacement_buffer = (neighbor.skin_fraction - 1.0) * min_skin;
    let ghost_cut = max_cutoff + 2.0 * displacement_buffer;
    neighbor.ghost_cutoff = ghost_cut;
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

    let whole_number_of_bins = domain.sub_length / neighbor.bin_min_size;
    let xi = whole_number_of_bins.x.floor() as i32;
    let yi = whole_number_of_bins.y.floor() as i32;
    let zi = whole_number_of_bins.z.floor() as i32;

    let actual_bin_size = Vector3::new(
        domain.sub_length.x / xi as f64,
        domain.sub_length.y / yi as f64,
        domain.sub_length.z / zi as f64,
    );

    // Stencil range: ceil(cutoff / bin_size) in each dimension
    let sx = (max_cutoff / actual_bin_size.x).ceil() as i32;
    let sy = (max_cutoff / actual_bin_size.y).ceil() as i32;
    let sz = (max_cutoff / actual_bin_size.z).ceil() as i32;

    // Ghost layers must match stencil range
    let nx = xi + 2 * sx;
    let ny = yi + 2 * sy;
    let nz = zi + 2 * sz;
    neighbor.bin_count = Vector3::new(nx, ny, nz);
    neighbor.bin_size = actual_bin_size;

    // Flat bin grid setup
    let total_cells = (nx * ny * nz) as usize;
    neighbor.bin_total_cells = total_cells;
    neighbor.bin_origin = domain.sub_domain_low - neighbor.bin_size.component_mul(&Vector3::new(sx as f64, sy as f64, sz as f64));

    // Precompute stencil offsets — only include cells whose minimum distance < cutoff
    let cutoff2 = max_cutoff * max_cutoff;
    neighbor.bin_stencil.clear();
    neighbor.bin_stencil_forward.clear();
    neighbor.bin_stencil_self = false;
    for dx in -sx..=sx {
        for dy in -sy..=sy {
            for dz in -sz..=sz {
                // Minimum distance from central cell to cell at offset (dx,dy,dz)
                let min_dx = (dx.unsigned_abs().saturating_sub(1)) as f64 * actual_bin_size.x;
                let min_dy = (dy.unsigned_abs().saturating_sub(1)) as f64 * actual_bin_size.y;
                let min_dz = (dz.unsigned_abs().saturating_sub(1)) as f64 * actual_bin_size.z;
                if min_dx * min_dx + min_dy * min_dy + min_dz * min_dz < cutoff2 {
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

    // Store box dimensions for minimum-image displacement check
    neighbor.pbc_box = [
        domain.boundaries_high.x - domain.boundaries_low.x,
        domain.boundaries_high.y - domain.boundaries_low.y,
        domain.boundaries_high.z - domain.boundaries_low.z,
    ];
    neighbor.pbc_flags = [domain.is_periodic.x, domain.is_periodic.y, domain.is_periodic.z];

    if comm.rank() == 0 {
        println!(
            "Neighbor: bins {}x{}x{} (with ghost layers), {} forward stencil cells",
            nx, ny, nz, neighbor.bin_stencil_forward.len()
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
    let (min_skin, max_skin) = atoms.skin[..nlocal]
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
    // displacement is small. Ghost ordering is deterministic in single-process
    // mode, so the neighbor list stays valid after wrapping.
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
pub fn decide_rebuild(mut atoms: ResMut<Atom>, neighbor: Res<Neighbor>, comm: Res<CommResource>) {
    if !atoms.communicate_only {
        return; // already going to do full rebuild
    }
    let local_needs = if needs_rebuild(&atoms, &neighbor) { 1.0 } else { 0.0 };
    // Any rank needing rebuild forces all ranks to rebuild
    let global_needs = comm.all_reduce_sum_f64(local_needs);
    if global_needs > 0.0 {
        atoms.communicate_only = false;
    }
}

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
        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let skin_fraction = neighbor.skin_fraction;
    for i in 0..neighbor.sweep_and_prune.len() {
        let index = neighbor.sweep_and_prune[i].0;
        let px = atoms.pos[index][0];
        let py = atoms.pos[index][1];
        let pz = atoms.pos[index][2];
        let r = atoms.skin[index];
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
            let r2 = atoms.skin[index2];
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
            if distance < (atoms.skin[i] + atoms.skin[j]) * neighbor.skin_fraction {
                neighbor.neighbor_list.push((i, j));
            }
        }
    }
    build_csr_from_pairs(&mut neighbor, nlocal);
}

pub fn sort_atoms_by_bin(mut atoms: ResMut<Atom>, mut neighbor: ResMut<Neighbor>, comm: Res<CommResource>, registry: Res<AtomDataRegistry>) {
    // Skip sort in MPI mode — atom reordering interacts with ghost exchange
    if comm.size() > 1 {
        return;
    }
    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 || neighbor.bin_total_cells == 0 || nlocal > atoms.pos.len() {
        return;
    }

    // Sort when neighbor rebuild is needed, or periodically based on sort_every
    neighbor.sort_counter += 1;
    let periodic_sort = neighbor.sort_every > 0 && neighbor.sort_counter >= neighbor.sort_every;
    let rebuild_needed = needs_rebuild(&atoms, &neighbor);
    if !periodic_sort && !rebuild_needed {
        return;
    }
    if periodic_sort {
        neighbor.sort_counter = 0;
    }

    let inv_bsx = 1.0 / neighbor.bin_size.x;
    let inv_bsy = 1.0 / neighbor.bin_size.y;
    let inv_bsz = 1.0 / neighbor.bin_size.z;
    let ny = neighbor.bin_count.y;
    let nz = neighbor.bin_count.z;
    let nx = neighbor.bin_count.x;

    let mut indices: Vec<(u32, usize)> = (0..nlocal)
        .map(|i| {
            let cx = ((atoms.pos[i][0] - neighbor.bin_origin.x) * inv_bsx).floor() as i32;
            let cy = ((atoms.pos[i][1] - neighbor.bin_origin.y) * inv_bsy).floor() as i32;
            let cz = ((atoms.pos[i][2] - neighbor.bin_origin.z) * inv_bsz).floor() as i32;
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
    if already_sorted {
        return;
    }

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
        // When sort was triggered by needs_rebuild, bin_neighbor_list will also rebuild
        // (same conditions still hold after permutation), so the stale CSR indices
        // will be replaced before any force computation uses them.
        // For periodic sort (needs_rebuild was false), we must force a rebuild since
        // CSR indices are stale but bin_neighbor_list wouldn't otherwise rebuild.
        if periodic_sort && !rebuild_needed {
            neighbor.last_build_pos.clear();
            atoms.communicate_only = false; // force full borders rebuild after sort
        }
    }
}

pub fn bin_neighbor_list(
    mut atoms: ResMut<Atom>,
    mut neighbor: ResMut<Neighbor>,
    _domain: Res<Domain>,
    _comm: Res<CommResource>,
) {
    let nlocal = atoms.nlocal as usize;
    let total = atoms.len();

    if !needs_rebuild(&atoms, &neighbor) {
        neighbor.steps_since_build += 1;
        // TEMP DISABLED for debugging: atoms.communicate_only = true;
        return;
    }
    // TEMP DISABLED for debugging: atoms.communicate_only = true;

    save_build_positions(&atoms, &mut neighbor);

    let ny = neighbor.bin_count.y;
    let nz = neighbor.bin_count.z;
    let nx = neighbor.bin_count.x;
    let total_cells = neighbor.bin_total_cells;
    let inv_bsx = 1.0 / neighbor.bin_size.x;
    let inv_bsy = 1.0 / neighbor.bin_size.y;
    let inv_bsz = 1.0 / neighbor.bin_size.z;
    let bin_ox = neighbor.bin_origin.x;
    let bin_oy = neighbor.bin_origin.y;
    let bin_oz = neighbor.bin_origin.z;

    // Step 1: Assign each atom to a bin cell (reuse persistent arrays)
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
                    let r2 = dx * dx + dy * dy + dz * dz;
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
                    let r2 = dx * dx + dy * dy + dz * dz;
                    if r2 < cutoff_sq {
                        let j = unsafe { *sorted_atoms.get_unchecked(m) };
                        push_index!(j);
                    }
                }
            }
        }
    } else {
        // Slow path: per-pair skin computation
        for i in 0..nlocal {
            offsets.push(nidx as u32);
            let my_cell = unsafe { *atom_cell.get_unchecked(i) } as usize;
            let pi = unsafe { *atoms.pos.get_unchecked(i) };
            let si = unsafe { *atoms.skin.get_unchecked(i) };

            if has_self {
                let self_start = unsafe { *atom_sorted_idx.get_unchecked(i) } as usize + 1;
                let end = unsafe { *bin_start.get_unchecked(my_cell + 1) } as usize;
                for m in self_start..end {
                    let j = unsafe { *sorted_atoms.get_unchecked(m) } as usize;
                    let pj = unsafe { *sorted_pos.get_unchecked(m) };
                    let dx = pj[0] - pi[0];
                    let dy = pj[1] - pi[1];
                    let dz = pj[2] - pi[2];
                    let r2 = dx * dx + dy * dy + dz * dz;
                    let cutoff = (si + unsafe { *atoms.skin.get_unchecked(j) }) * skin_fraction;
                    if r2 < cutoff * cutoff {
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
                    let r2 = dx * dx + dy * dy + dz * dz;
                    let cutoff = (si + unsafe { *atoms.skin.get_unchecked(j) }) * skin_fraction;
                    if r2 < cutoff * cutoff {
                        push_index!(j as u32);
                    }
                }
            }
        }
    }
    // SAFETY: nidx elements were written via raw pointer; all < capacity.
    unsafe { indices.set_len(nidx) };
    offsets.push(nidx as u32);

    // Sort each atom's neighbor indices by j for cache-friendly access in force loops.
    // Sequential j access improves prefetching for atoms.pos[j] and atoms.force[j].
    for i in 0..nlocal {
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        indices[s..e].sort_unstable();
    }

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
    use nalgebra::Vector3;

    fn push_atom(atom: &mut Atom, tag: u32, pos: Vector3<f64>, radius: f64) {
        atom.push_test_atom(tag, pos, radius, 1.0);
    }

    #[test]
    fn brute_force_finds_close_pair() {
        let mut app = App::new();
        let mut atom = Atom::new();
        push_atom(&mut atom, 0, Vector3::new(0.0, 0.0, 0.0), 0.5);
        push_atom(&mut atom, 1, Vector3::new(0.5, 0.0, 0.0), 0.5);
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
        push_atom(&mut atom, 0, Vector3::new(0.0, 0.0, 0.0), 0.1);
        push_atom(&mut atom, 1, Vector3::new(5.0, 0.0, 0.0), 0.1);
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
        push_atom(&mut atom, 0, Vector3::new(0.5, 0.5, 0.5), 0.5);
        push_atom(&mut atom, 1, Vector3::new(1.0, 0.5, 0.5), 0.5);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut domain = mddem_core::Domain::new();
        domain.sub_domain_low = Vector3::new(0.0, 0.0, 0.0);
        domain.sub_domain_high = Vector3::new(2.0, 2.0, 2.0);
        domain.sub_length = Vector3::new(2.0, 2.0, 2.0);

        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = 1.0;
        neighbor.bin_min_size = 1.0;

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(domain);
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
        push_atom(&mut atom, 0, Vector3::new(0.0, 0.0, 0.0), 0.5);
        push_atom(&mut atom, 1, Vector3::new(0.5, 0.0, 0.0), 0.5);
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
}

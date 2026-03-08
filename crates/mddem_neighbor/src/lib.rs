use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use mddem_core::{Atom, CommResource, Config, Domain};

fn default_one_f64() -> f64 {
    1.0
}
fn default_zero_usize() -> usize {
    0
}
fn default_true() -> bool {
    true
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NeighborConfig {
    #[serde(default = "default_one_f64")]
    pub skin_fraction: f64,
    #[serde(default = "default_one_f64")]
    pub bin_size: f64,
    /// Rebuild every N steps (0 = displacement-based only)
    #[serde(default = "default_zero_usize")]
    pub every: usize,
    /// When true and every > 0, also check displacement threshold (like LAMMPS "check yes")
    #[serde(default = "default_true")]
    pub check: bool,
}

impl Default for NeighborConfig {
    fn default() -> Self {
        NeighborConfig {
            skin_fraction: 1.0,
            bin_size: 1.0,
            every: 0,
            check: true,
        }
    }
}

pub enum NeighborStyle {
    BruteForce,
    SweepAndPrune,
    Bin,
}

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
    pub last_build_pos_x: Vec<f64>,
    pub last_build_pos_y: Vec<f64>,
    pub last_build_pos_z: Vec<f64>,
    pub last_build_tags: Vec<u32>,
    pub steps_since_build: usize,
    pub last_build_total: usize,
    pub bin_stencil: Vec<i32>,    // flat offsets for neighbor cell stencil (full)
    pub bin_stencil_forward: Vec<i32>, // offsets > 0 (half stencil for local-local dedup)
    pub bin_stencil_self: bool,        // whether self-cell (offset 0) is in stencil
    pub bin_origin: Vector3<f64>, // lower-left corner of bin grid (sub_domain_low - bin_size)
    pub bin_total_cells: usize,   // nx * ny * nz (including ghost layers)
    pub every: usize,             // rebuild every N steps (0 = displacement only)
    pub check: bool,              // also check displacement when every > 0
    pub ghost_cutoff: f64,        // max distance for ghost atom communication
    // Persistent scratch buffers (reused across rebuilds)
    pub scratch_atom_cell: Vec<u32>,
    pub scratch_bin_count: Vec<u32>,
    pub scratch_bin_start: Vec<u32>,
    pub scratch_write_pos: Vec<u32>,
    pub scratch_sorted_atoms: Vec<u32>,
    pub scratch_sorted_x: Vec<f64>,
    pub scratch_sorted_y: Vec<f64>,
    pub scratch_sorted_z: Vec<f64>,
    pub scratch_r2: Vec<f64>,
}

impl Default for Neighbor {
    fn default() -> Self {
        Self::new()
    }
}

impl Neighbor {
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
            last_build_pos_x: Vec::new(),
            last_build_pos_y: Vec::new(),
            last_build_pos_z: Vec::new(),
            last_build_tags: Vec::new(),
            steps_since_build: 0,
            last_build_total: 0,
            bin_stencil: Vec::new(),
            bin_stencil_forward: Vec::new(),
            bin_stencil_self: false,
            bin_origin: Vector3::new(0.0, 0.0, 0.0),
            bin_total_cells: 0,
            every: 0,
            check: true,
            ghost_cutoff: 0.0,
            scratch_atom_cell: Vec::new(),
            scratch_bin_count: Vec::new(),
            scratch_bin_start: Vec::new(),
            scratch_write_pos: Vec::new(),
            scratch_sorted_atoms: Vec::new(),
            scratch_sorted_x: Vec::new(),
            scratch_sorted_y: Vec::new(),
            scratch_sorted_z: Vec::new(),
            scratch_r2: Vec::new(),
        }
    }
}

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
check = true"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<NeighborConfig>(app, "neighbor");

        app.add_resource(Neighbor::new())
            .add_setup_system(neighbor_read_input, ScheduleSetupSet::Setup)
            .add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
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
                        .before("single_process_borders")
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
    neighbor.ghost_cutoff = max_cutoff;
    domain.ghost_cutoff = max_cutoff;
    if comm.rank() == 0 {
        println!("Neighbor: ghost_cutoff={:.4}", max_cutoff);
    }
    // Single-process: bin_size >= cutoff gives stencil range=1, 13 forward cells
    // Multi-process: use cutoff/2 so subdomains have enough bins for correct stencil
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
/// Only saves local atoms — ghost atoms are ephemeral and must not affect rebuild decisions.
fn save_build_positions(atoms: &Atom, neighbor: &mut Neighbor) {
    let nlocal = atoms.nlocal as usize;
    neighbor.last_build_pos_x.resize(nlocal, 0.0);
    neighbor.last_build_pos_y.resize(nlocal, 0.0);
    neighbor.last_build_pos_z.resize(nlocal, 0.0);
    neighbor.last_build_pos_x[..nlocal].copy_from_slice(&atoms.pos_x[..nlocal]);
    neighbor.last_build_pos_y[..nlocal].copy_from_slice(&atoms.pos_y[..nlocal]);
    neighbor.last_build_pos_z[..nlocal].copy_from_slice(&atoms.pos_z[..nlocal]);
    neighbor.last_build_tags.clear();
    neighbor.last_build_tags.extend_from_slice(&atoms.tag[..nlocal]);
    neighbor.last_build_total = atoms.len();
    neighbor.steps_since_build = 0;
}

/// Helper: check displacement threshold
fn displacement_exceeded(atoms: &Atom, neighbor: &Neighbor) -> bool {
    let nlocal = atoms.nlocal as usize;
    let min_r = atoms.skin[..nlocal]
        .iter()
        .cloned()
        .fold(f64::MAX, f64::min);
    let half_skin = (neighbor.skin_fraction - 1.0) * min_r * 0.5;
    let threshold_sq = half_skin * half_skin;
    for idx in 0..neighbor.last_build_pos_x.len() {
        let dx = atoms.pos_x[idx] - neighbor.last_build_pos_x[idx];
        let dy = atoms.pos_y[idx] - neighbor.last_build_pos_y[idx];
        let dz = atoms.pos_z[idx] - neighbor.last_build_pos_z[idx];
        if dx * dx + dy * dy + dz * dz > threshold_sq {
            return true;
        }
    }
    false
}

/// Helper: check if a rebuild is needed.
/// Compares against local atoms only — ghost atoms are recreated every step.
fn needs_rebuild(atoms: &Atom, neighbor: &Neighbor) -> bool {
    let nlocal = atoms.nlocal as usize;
    // Always rebuild on first call, local atom count change, or total count change
    // (ghost atoms are recreated each step, so stale indices become invalid)
    if neighbor.last_build_pos_x.len() != nlocal || nlocal == 0 || neighbor.last_build_total != atoms.len() {
        return true;
    }
    let tags_unchanged = atoms.tag[..nlocal]
        .iter()
        .zip(neighbor.last_build_tags.iter())
        .all(|(t, lt)| t == lt);
    if !tags_unchanged {
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

pub fn sweep_and_prune_neighbor_list(
    atoms: Res<Atom>,
    mut neighbor: ResMut<Neighbor>,
    _domain: Res<Domain>,
    _comm: Res<CommResource>,
) {
    if !needs_rebuild(&atoms, &neighbor) {
        neighbor.steps_since_build += 1;
        return;
    }

    save_build_positions(&atoms, &mut neighbor);
    neighbor.sweep_and_prune.clear();
    neighbor.neighbor_list.clear();

    for j in 0..atoms.len() {
        neighbor.sweep_and_prune.push((j, atoms.pos_x[j]));
    }
    neighbor
        .sweep_and_prune
        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let skin_fraction = neighbor.skin_fraction;
    for i in 0..neighbor.sweep_and_prune.len() {
        let index = neighbor.sweep_and_prune[i].0;
        let px = atoms.pos_x[index];
        let py = atoms.pos_y[index];
        let pz = atoms.pos_z[index];
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
            let dx = atoms.pos_x[index2] - px;
            let dy = atoms.pos_y[index2] - py;
            let dz = atoms.pos_z[index2] - pz;
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
            let dx = atoms.pos_x[j] - atoms.pos_x[i];
            let dy = atoms.pos_y[j] - atoms.pos_y[i];
            let dz = atoms.pos_z[j] - atoms.pos_z[i];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            if distance < (atoms.skin[i] + atoms.skin[j]) * neighbor.skin_fraction {
                neighbor.neighbor_list.push((i, j));
            }
        }
    }
    build_csr_from_pairs(&mut neighbor, nlocal);
}

fn permute_vec_f64(v: &mut Vec<f64>, perm: &[usize], n: usize) {
    let scratch: Vec<f64> = perm.iter().map(|&p| v[p]).collect();
    v[..n].copy_from_slice(&scratch);
}

fn permute_vec_u32(v: &mut Vec<u32>, perm: &[usize], n: usize) {
    let scratch: Vec<u32> = perm.iter().map(|&p| v[p]).collect();
    v[..n].copy_from_slice(&scratch);
}

fn permute_vec_i32(v: &mut Vec<i32>, perm: &[usize], n: usize) {
    let scratch: Vec<i32> = perm.iter().map(|&p| v[p]).collect();
    v[..n].copy_from_slice(&scratch);
}

fn permute_vec_bool(v: &mut Vec<bool>, perm: &[usize], n: usize) {
    let scratch: Vec<bool> = perm.iter().map(|&p| v[p]).collect();
    v[..n].copy_from_slice(&scratch);
}

fn apply_permutation_to_atoms(atoms: &mut Atom, perm: &[usize], n: usize) {
    permute_vec_f64(&mut atoms.pos_x, perm, n);
    permute_vec_f64(&mut atoms.pos_y, perm, n);
    permute_vec_f64(&mut atoms.pos_z, perm, n);
    permute_vec_f64(&mut atoms.vel_x, perm, n);
    permute_vec_f64(&mut atoms.vel_y, perm, n);
    permute_vec_f64(&mut atoms.vel_z, perm, n);
    permute_vec_f64(&mut atoms.force_x, perm, n);
    permute_vec_f64(&mut atoms.force_y, perm, n);
    permute_vec_f64(&mut atoms.force_z, perm, n);
    permute_vec_f64(&mut atoms.torque_x, perm, n);
    permute_vec_f64(&mut atoms.torque_y, perm, n);
    permute_vec_f64(&mut atoms.torque_z, perm, n);
    permute_vec_f64(&mut atoms.omega_x, perm, n);
    permute_vec_f64(&mut atoms.omega_y, perm, n);
    permute_vec_f64(&mut atoms.omega_z, perm, n);
    permute_vec_f64(&mut atoms.ang_mom_x, perm, n);
    permute_vec_f64(&mut atoms.ang_mom_y, perm, n);
    permute_vec_f64(&mut atoms.ang_mom_z, perm, n);
    permute_vec_f64(&mut atoms.mass, perm, n);
    permute_vec_f64(&mut atoms.skin, perm, n);
    permute_vec_u32(&mut atoms.tag, perm, n);
    permute_vec_u32(&mut atoms.atom_type, perm, n);
    permute_vec_i32(&mut atoms.origin_index, perm, n);
    permute_vec_bool(&mut atoms.is_ghost, perm, n);
    permute_vec_bool(&mut atoms.has_ghost, perm, n);
    permute_vec_bool(&mut atoms.is_collision, perm, n);
    // Quaternion: Vec<UnitQuaternion> — use clone
    let scratch_q: Vec<_> = perm.iter().map(|&p| atoms.quaterion[p].clone()).collect();
    atoms.quaterion[..n].clone_from_slice(&scratch_q);
}

pub fn sort_atoms_by_bin(mut atoms: ResMut<Atom>, neighbor: Res<Neighbor>, comm: Res<CommResource>) {
    // Skip sort in MPI mode — atom reordering interacts with ghost exchange
    if comm.size() > 1 {
        return;
    }
    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 || neighbor.bin_total_cells == 0 || nlocal > atoms.pos_x.len() {
        return;
    }

    // Only sort when a neighbor rebuild will happen
    if !needs_rebuild(&atoms, &neighbor) {
        return;
    }

    let inv_bsx = 1.0 / neighbor.bin_size.x;
    let inv_bsy = 1.0 / neighbor.bin_size.y;
    let inv_bsz = 1.0 / neighbor.bin_size.z;
    let ny = neighbor.bin_count.y;
    let nz = neighbor.bin_count.z;
    let nx = neighbor.bin_count.x;

    let mut indices: Vec<(u32, usize)> = (0..nlocal)
        .map(|i| {
            let cx = ((atoms.pos_x[i] - neighbor.bin_origin.x) * inv_bsx).floor() as i32;
            let cy = ((atoms.pos_y[i] - neighbor.bin_origin.y) * inv_bsy).floor() as i32;
            let cz = ((atoms.pos_z[i] - neighbor.bin_origin.z) * inv_bsz).floor() as i32;
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

    apply_permutation_to_atoms(&mut atoms, &perm, nlocal);
}

pub fn bin_neighbor_list(
    atoms: Res<Atom>,
    mut neighbor: ResMut<Neighbor>,
    _domain: Res<Domain>,
    _comm: Res<CommResource>,
) {
    let nlocal = atoms.nlocal as usize;
    let total = atoms.len();

    if !needs_rebuild(&atoms, &neighbor) {
        neighbor.steps_since_build += 1;
        return;
    }

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

    // Step 1: Build CSR bin structure for ALL atoms (local + ghost)
    // Use persistent scratch buffers to avoid repeated allocations
    let mut atom_cell = std::mem::take(&mut neighbor.scratch_atom_cell);
    let mut bin_count_arr = std::mem::take(&mut neighbor.scratch_bin_count);
    let mut bin_start = std::mem::take(&mut neighbor.scratch_bin_start);
    let mut sorted_atoms = std::mem::take(&mut neighbor.scratch_sorted_atoms);
    let mut write_pos = std::mem::take(&mut neighbor.scratch_write_pos);
    let mut sorted_x = std::mem::take(&mut neighbor.scratch_sorted_x);
    let mut sorted_y = std::mem::take(&mut neighbor.scratch_sorted_y);
    let mut sorted_z = std::mem::take(&mut neighbor.scratch_sorted_z);

    atom_cell.resize(total, 0);
    bin_count_arr.resize(total_cells, 0);
    bin_count_arr.fill(0);

    for i in 0..total {
        let cx = ((atoms.pos_x[i] - bin_ox) * inv_bsx).floor() as i32;
        let cy = ((atoms.pos_y[i] - bin_oy) * inv_bsy).floor() as i32;
        let cz = ((atoms.pos_z[i] - bin_oz) * inv_bsz).floor() as i32;
        let cx = cx.clamp(0, nx - 1);
        let cy = cy.clamp(0, ny - 1);
        let cz = cz.clamp(0, nz - 1);
        let cell = (cx * ny * nz + cy * nz + cz) as u32;
        atom_cell[i] = cell;
        bin_count_arr[cell as usize] += 1;
    }

    // CSR offsets: bin_start[c]..bin_start[c+1]
    bin_start.resize(total_cells + 1, 0);
    bin_start[0] = 0;
    for c in 0..total_cells {
        bin_start[c + 1] = bin_start[c] + bin_count_arr[c];
    }

    // Place atoms into sorted order
    sorted_atoms.resize(total, 0);
    write_pos.resize(total_cells, 0);
    write_pos.copy_from_slice(&bin_start[..total_cells]);
    for i in 0..total {
        let c = atom_cell[i] as usize;
        sorted_atoms[write_pos[c] as usize] = i as u32;
        write_pos[c] += 1;
    }

    // Step 2: Build sorted position arrays for cache-friendly inner loop
    sorted_x.resize(total, 0.0);
    sorted_y.resize(total, 0.0);
    sorted_z.resize(total, 0.0);
    for (m, &ai) in sorted_atoms.iter().enumerate() {
        let a = ai as usize;
        sorted_x[m] = atoms.pos_x[a];
        sorted_y[m] = atoms.pos_y[a];
        sorted_z[m] = atoms.pos_z[a];
    }

    // Step 3: Build CSR neighbor lists using forward stencil.
    // Each pair is found exactly once: self cell uses j > i dedup,
    // forward cells have positive offset so each pair appears once.
    // Ghost atoms in backward cells are found by the neighboring proc's
    // forward stencil; forces return via reverse_comm.
    let stencil_other = neighbor.bin_stencil_forward.clone();
    let has_self = neighbor.bin_stencil_self;
    let skin_fraction = neighbor.skin_fraction;
    let skin_fraction2 = skin_fraction * skin_fraction;
    let total_cells_i32 = total_cells as i32;

    let uniform_skin = total > 0 && atoms.skin[..total].iter().all(|&s| s == atoms.skin[0]);
    let cutoff2_uniform = if uniform_skin {
        let c = 2.0 * atoms.skin[0] * skin_fraction;
        c * c
    } else {
        0.0
    };

    let prev_count = neighbor.neighbor_indices.len();
    neighbor.neighbor_offsets.clear();
    neighbor.neighbor_indices.clear();
    neighbor.neighbor_offsets.reserve(nlocal + 1);
    neighbor.neighbor_indices.reserve(prev_count + prev_count / 4);

    // Scratch buffer for vectorizable distance computation
    let mut scratch_r2 = std::mem::take(&mut neighbor.scratch_r2);

    if uniform_skin {
        for i in 0..nlocal {
            let off = neighbor.neighbor_indices.len() as u32;
            neighbor.neighbor_offsets.push(off);

            let my_cell = atom_cell[i] as i32;
            let xi = atoms.pos_x[i];
            let yi = atoms.pos_y[i];
            let zi = atoms.pos_z[i];

            // Self cell: j > i dedup (ghost j always > i since j >= nlocal > i)
            if has_self {
                let c = my_cell as usize;
                let start = bin_start[c] as usize;
                let end = bin_start[c + 1] as usize;
                let count = end - start;

                // Phase 1: compute all r2 values (auto-vectorizable)
                scratch_r2.resize(count, 0.0);
                for k in 0..count {
                    let m = start + k;
                    let dx = sorted_x[m] - xi;
                    let dy = sorted_y[m] - yi;
                    let dz = sorted_z[m] - zi;
                    scratch_r2[k] = dx * dx + dy * dy + dz * dz;
                }

                // Phase 2: collect passing neighbors (sequential)
                for k in 0..count {
                    let j = sorted_atoms[start + k] as usize;
                    if j <= i { continue; }
                    if scratch_r2[k] < cutoff2_uniform {
                        neighbor.neighbor_indices.push(j as u32);
                    }
                }
            }

            // Other stencil cells: in single-proc forward-only (no dedup needed);
            // in multi-proc full stencil, dedup local-local pairs with j > i
            for &offset in &stencil_other {
                let nc = my_cell + offset;
                if nc < 0 || nc >= total_cells_i32 { continue; }
                let c = nc as usize;
                let start = bin_start[c] as usize;
                let end = bin_start[c + 1] as usize;
                let count = end - start;

                // Phase 1: compute all r2 values (auto-vectorizable)
                scratch_r2.resize(count, 0.0);
                for k in 0..count {
                    let m = start + k;
                    let dx = sorted_x[m] - xi;
                    let dy = sorted_y[m] - yi;
                    let dz = sorted_z[m] - zi;
                    scratch_r2[k] = dx * dx + dy * dy + dz * dz;
                }

                // Phase 2: collect passing neighbors (sequential)
                // Forward stencil: all atoms included (no dedup needed)
                for k in 0..count {
                    if scratch_r2[k] < cutoff2_uniform {
                        let j = sorted_atoms[start + k] as usize;
                        neighbor.neighbor_indices.push(j as u32);
                    }
                }
            }
        }
    } else {
        for i in 0..nlocal {
            let off = neighbor.neighbor_indices.len() as u32;
            neighbor.neighbor_offsets.push(off);

            let my_cell = atom_cell[i] as i32;
            let xi = atoms.pos_x[i];
            let yi = atoms.pos_y[i];
            let zi = atoms.pos_z[i];
            let si = atoms.skin[i];

            if has_self {
                let c = my_cell as usize;
                for m in bin_start[c] as usize..bin_start[c + 1] as usize {
                    let j = sorted_atoms[m] as usize;
                    if j <= i { continue; }
                    let dx = sorted_x[m] - xi;
                    let dy = sorted_y[m] - yi;
                    let dz = sorted_z[m] - zi;
                    let r2 = dx * dx + dy * dy + dz * dz;
                    let cutoff = si + atoms.skin[j];
                    if r2 < cutoff * cutoff * skin_fraction2 {
                        neighbor.neighbor_indices.push(j as u32);
                    }
                }
            }

            for &offset in &stencil_other {
                let nc = my_cell + offset;
                if nc < 0 || nc >= total_cells_i32 { continue; }
                let c = nc as usize;
                for m in bin_start[c] as usize..bin_start[c + 1] as usize {
                    let j = sorted_atoms[m] as usize;
                    let dx = sorted_x[m] - xi;
                    let dy = sorted_y[m] - yi;
                    let dz = sorted_z[m] - zi;
                    let r2 = dx * dx + dy * dy + dz * dz;
                    let cutoff = si + atoms.skin[j];
                    if r2 < cutoff * cutoff * skin_fraction2 {
                        neighbor.neighbor_indices.push(j as u32);
                    }
                }
            }
        }
    }
    let final_off = neighbor.neighbor_indices.len() as u32;
    neighbor.neighbor_offsets.push(final_off);

    // Return scratch buffers
    neighbor.scratch_r2 = scratch_r2;

    // Return scratch buffers for reuse
    neighbor.scratch_atom_cell = atom_cell;
    neighbor.scratch_bin_count = bin_count_arr;
    neighbor.scratch_bin_start = bin_start;
    neighbor.scratch_write_pos = write_pos;
    neighbor.scratch_sorted_atoms = sorted_atoms;
    neighbor.scratch_sorted_x = sorted_x;
    neighbor.scratch_sorted_y = sorted_y;
    neighbor.scratch_sorted_z = sorted_z;
}

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::Atom;
    use nalgebra::{UnitQuaternion, Vector3};

    fn push_atom(atom: &mut Atom, tag: u32, pos: Vector3<f64>, radius: f64) {
        atom.tag.push(tag);
        atom.atom_type.push(0);
        atom.origin_index.push(0);
        atom.pos_x.push(pos.x);
        atom.pos_y.push(pos.y);
        atom.pos_z.push(pos.z);
        atom.vel_x.push(0.0);
        atom.vel_y.push(0.0);
        atom.vel_z.push(0.0);
        atom.force_x.push(0.0);
        atom.force_y.push(0.0);
        atom.force_z.push(0.0);
        atom.torque_x.push(0.0);
        atom.torque_y.push(0.0);
        atom.torque_z.push(0.0);
        atom.mass.push(1.0);
        atom.skin.push(radius);
        atom.is_ghost.push(false);
        atom.has_ghost.push(false);
        atom.is_collision.push(false);
        atom.quaterion.push(UnitQuaternion::identity());
        atom.omega_x.push(0.0);
        atom.omega_y.push(0.0);
        atom.omega_z.push(0.0);
        atom.ang_mom_x.push(0.0);
        atom.ang_mom_y.push(0.0);
        atom.ang_mom_z.push(0.0);
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

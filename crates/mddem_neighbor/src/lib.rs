use serde::{Deserialize, Serialize};
use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;

use mddem_core::{Config, Atom, CommResource, Domain};

fn default_one_f64() -> f64 { 1.0 }

#[derive(Serialize, Deserialize, Clone)]
pub struct NeighborConfig {
    #[serde(default = "default_one_f64")]
    pub skin_fraction: f64,
    #[serde(default = "default_one_f64")]
    pub bin_size: f64,
}

impl Default for NeighborConfig {
    fn default() -> Self {
        NeighborConfig { skin_fraction: 1.0, bin_size: 1.0 }
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
    pub sweep_and_prune: Vec<(usize, f64)>,
    pub bin_min_size: f64,
    pub bin_size: Vector3<f64>,
    pub bin_count: Vector3<i32>,
    pub last_build_pos: Vec<Vector3<f64>>,
    pub last_build_tags: Vec<u32>,
    pub steps_since_build: usize,
    // Flat bin arrays for GPU-ready linked-list cell storage
    pub bin_head: Vec<i32>,       // length = total_cells; -1 = empty
    pub bin_next: Vec<i32>,       // length = max atoms; next atom in same cell, -1 = end
    pub bin_stencil: Vec<i32>,    // 27 flat offsets for neighbor cell stencil
    pub bin_origin: Vector3<f64>, // lower-left corner of bin grid (sub_domain_low - bin_size)
    pub bin_total_cells: usize,   // nx * ny * nz (including ghost layers)
}

impl Neighbor {
    pub fn new() -> Self {
        Neighbor {
            skin_fraction: 1.0, neighbor_list: Vec::new(),
            sweep_and_prune: Vec::new(), bin_min_size: 1.0,
            bin_size: Vector3::new(1.0, 1.0, 1.0), bin_count: Vector3::new(1, 1, 1),
            last_build_pos: Vec::new(), last_build_tags: Vec::new(), steps_since_build: 0,
            bin_head: Vec::new(), bin_next: Vec::new(), bin_stencil: Vec::new(),
            bin_origin: Vector3::new(0.0, 0.0, 0.0), bin_total_cells: 0,
        }
    }
}

pub struct NeighborPlugin {
    pub style: NeighborStyle,
}

impl Plugin for NeighborPlugin {
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
                app.add_update_system(bin_neighbor_list, ScheduleSet::Neighbor);
            }
        }
    }
}

pub fn neighbor_read_input(config: Res<NeighborConfig>, mut neighbor: ResMut<Neighbor>, comm: Res<CommResource>) {
    neighbor.skin_fraction = config.skin_fraction;
    neighbor.bin_min_size = config.bin_size;
    if comm.rank() == 0 { println!("Neighbor: skin_fraction={} bin_size={}", config.skin_fraction, config.bin_size); }
}

pub fn neighbor_setup(mut neighbor: ResMut<Neighbor>, domain: Res<Domain>) {
    let whole_number_of_bins = domain.sub_length / neighbor.bin_min_size;
    let xi = whole_number_of_bins.x.floor() as i32;
    let yi = whole_number_of_bins.y.floor() as i32;
    let zi = whole_number_of_bins.z.floor() as i32;
    // +2 for one ghost layer on each side
    let nx = xi + 2;
    let ny = yi + 2;
    let nz = zi + 2;
    neighbor.bin_count = Vector3::new(nx, ny, nz);
    neighbor.bin_size = Vector3::new(
        domain.sub_length.x / xi as f64,
        domain.sub_length.y / yi as f64,
        domain.sub_length.z / zi as f64,
    );

    // Flat bin grid setup
    let total_cells = (nx * ny * nz) as usize;
    neighbor.bin_total_cells = total_cells;
    neighbor.bin_head = vec![-1; total_cells];
    neighbor.bin_origin = domain.sub_domain_low - neighbor.bin_size;

    // Precompute 27 stencil offsets
    neighbor.bin_stencil.clear();
    for dx in -1..=1_i32 {
        for dy in -1..=1_i32 {
            for dz in -1..=1_i32 {
                let offset = dx * ny * nz + dy * nz + dz;
                neighbor.bin_stencil.push(offset);
            }
        }
    }
}

/// Helper: save current positions for displacement-based rebuild check
fn save_build_positions(atoms: &Atom, neighbor: &mut Neighbor) {
    neighbor.last_build_pos.clear();
    for i in 0..atoms.len() {
        neighbor.last_build_pos.push(Vector3::new(atoms.pos_x[i], atoms.pos_y[i], atoms.pos_z[i]));
    }
    neighbor.last_build_tags.clear();
    neighbor.last_build_tags.extend_from_slice(&atoms.tag);
    neighbor.steps_since_build = 0;
}

/// Helper: check if a rebuild is needed based on displacement
fn needs_rebuild(atoms: &Atom, neighbor: &Neighbor) -> bool {
    let total = atoms.len();
    let nlocal = atoms.nlocal as usize;
    if neighbor.last_build_pos.len() != total || total == 0 {
        return true;
    }
    let tags_unchanged = atoms.tag.iter().zip(neighbor.last_build_tags.iter()).all(|(t, lt)| t == lt);
    if !tags_unchanged {
        return true;
    }
    let min_r = atoms.skin[..nlocal].iter().cloned().fold(f64::MAX, f64::min);
    let half_skin = (neighbor.skin_fraction - 1.0) * min_r * 0.5;
    let threshold_sq = half_skin * half_skin;
    for (idx, last) in neighbor.last_build_pos.iter().enumerate() {
        let dx = atoms.pos_x[idx] - last.x;
        let dy = atoms.pos_y[idx] - last.y;
        let dz = atoms.pos_z[idx] - last.z;
        if dx*dx + dy*dy + dz*dz > threshold_sq {
            return true;
        }
    }
    false
}

pub fn sweep_and_prune_neighbor_list(atoms: Res<Atom>, mut neighbor: ResMut<Neighbor>, _domain: Res<Domain>, _comm: Res<CommResource>) {
    if !needs_rebuild(&atoms, &neighbor) {
        neighbor.steps_since_build += 1;
        return;
    }

    save_build_positions(&atoms, &mut neighbor);
    neighbor.sweep_and_prune.clear();
    neighbor.neighbor_list.clear();

    for j in 0..atoms.len() { neighbor.sweep_and_prune.push((j, atoms.pos_x[j])); }
    neighbor.sweep_and_prune.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let skin_fraction = neighbor.skin_fraction;
    for i in 0..neighbor.sweep_and_prune.len() {
        let index = neighbor.sweep_and_prune[i].0;
        let px = atoms.pos_x[index];
        let py = atoms.pos_y[index];
        let pz = atoms.pos_z[index];
        let r = atoms.skin[index];
        for j in (i+1)..neighbor.sweep_and_prune.len() {
            if (neighbor.sweep_and_prune[j].1 - px) > (r * 2.0 * skin_fraction) { break; }
            let index2 = neighbor.sweep_and_prune[j].0;
            if atoms.tag[index] == atoms.tag[index2] || (atoms.is_ghost[index] && atoms.is_ghost[index2]) { continue; }
            let r2 = atoms.skin[index2];
            let dx = atoms.pos_x[index2] - px;
            let dy = atoms.pos_y[index2] - py;
            let dz = atoms.pos_z[index2] - pz;
            let distance = (dx*dx + dy*dy + dz*dz).sqrt();
            if distance < (r + r2)*skin_fraction { neighbor.neighbor_list.push((index, index2)); }
        }
    }
}

pub fn brute_force_neighbor_list(atoms: Res<Atom>, mut neighbor: ResMut<Neighbor>) {
    neighbor.neighbor_list.clear();
    let nlocal = atoms.len() - atoms.nghost as usize;
    for i in 0..nlocal {
        for j in (i+1)..atoms.len() {
            if atoms.tag[i] == atoms.tag[j] { continue; }
            let dx = atoms.pos_x[j] - atoms.pos_x[i];
            let dy = atoms.pos_y[j] - atoms.pos_y[i];
            let dz = atoms.pos_z[j] - atoms.pos_z[i];
            let distance = (dx*dx + dy*dy + dz*dz).sqrt();
            if distance < (atoms.skin[i] + atoms.skin[j])*neighbor.skin_fraction { neighbor.neighbor_list.push((i, j)); }
        }
    }
}

pub fn bin_neighbor_list(atoms: Res<Atom>, mut neighbor: ResMut<Neighbor>, _domain: Res<Domain>, _comm: Res<CommResource>) {
    let nlocal = atoms.nlocal as usize;
    let total = atoms.len();

    if !needs_rebuild(&atoms, &neighbor) {
        neighbor.steps_since_build += 1;
        return;
    }

    save_build_positions(&atoms, &mut neighbor);
    neighbor.neighbor_list.clear();

    let ny = neighbor.bin_count.y;
    let nz = neighbor.bin_count.z;
    let nx = neighbor.bin_count.x;

    // Step 1: Clear bin_head and resize bin_next
    for h in neighbor.bin_head.iter_mut() { *h = -1; }
    neighbor.bin_next.resize(total, -1);

    // Step 2: Assign atoms to bins via linked list
    for i in 0..total {
        let rx = atoms.pos_x[i] - neighbor.bin_origin.x;
        let ry = atoms.pos_y[i] - neighbor.bin_origin.y;
        let rz = atoms.pos_z[i] - neighbor.bin_origin.z;
        let cx = ((rx / neighbor.bin_size.x).floor() as i32).clamp(0, nx - 1);
        let cy = ((ry / neighbor.bin_size.y).floor() as i32).clamp(0, ny - 1);
        let cz = ((rz / neighbor.bin_size.z).floor() as i32).clamp(0, nz - 1);
        let cell = (cx * ny * nz + cy * nz + cz) as usize;
        neighbor.bin_next[i] = neighbor.bin_head[cell];
        neighbor.bin_head[cell] = i as i32;
    }

    // Step 3: Build pairs — iterate local atoms, check 27 stencil neighbors
    let stencil = neighbor.bin_stencil.clone();
    let skin_fraction = neighbor.skin_fraction;
    for i in 0..nlocal {
        let rx = atoms.pos_x[i] - neighbor.bin_origin.x;
        let ry = atoms.pos_y[i] - neighbor.bin_origin.y;
        let rz = atoms.pos_z[i] - neighbor.bin_origin.z;
        let cx = ((rx / neighbor.bin_size.x).floor() as i32).clamp(0, nx - 1);
        let cy = ((ry / neighbor.bin_size.y).floor() as i32).clamp(0, ny - 1);
        let cz = ((rz / neighbor.bin_size.z).floor() as i32).clamp(0, nz - 1);
        let my_cell = cx * ny * nz + cy * nz + cz;

        for &offset in &stencil {
            let neighbor_cell = my_cell + offset;
            if neighbor_cell < 0 || neighbor_cell >= neighbor.bin_total_cells as i32 { continue; }
            let mut j = neighbor.bin_head[neighbor_cell as usize];
            while j >= 0 {
                let ju = j as usize;
                if ju != i
                    && !(atoms.is_ghost[i] && atoms.is_ghost[ju])
                    && atoms.tag[i] != atoms.tag[ju]
                    // Avoid duplicate pairs: for atoms in the same or later cells, require j > i
                    && (my_cell != neighbor_cell || ju > i)
                {
                    let dx = atoms.pos_x[ju] - atoms.pos_x[i];
                    let dy = atoms.pos_y[ju] - atoms.pos_y[i];
                    let dz = atoms.pos_z[ju] - atoms.pos_z[i];
                    let distance = (dx*dx + dy*dy + dz*dz).sqrt();
                    if distance < (atoms.skin[i] + atoms.skin[ju]) * skin_fraction {
                        neighbor.neighbor_list.push((i, ju));
                    }
                }
                j = neighbor.bin_next[ju];
            }
        }
    }
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
        atom.pos_x.push(pos.x); atom.pos_y.push(pos.y); atom.pos_z.push(pos.z);
        atom.vel_x.push(0.0); atom.vel_y.push(0.0); atom.vel_z.push(0.0);
        atom.force_x.push(0.0); atom.force_y.push(0.0); atom.force_z.push(0.0);
        atom.torque_x.push(0.0); atom.torque_y.push(0.0); atom.torque_z.push(0.0);
        atom.mass.push(1.0);
        atom.skin.push(radius);
        atom.is_ghost.push(false);
        atom.has_ghost.push(false);
        atom.is_collision.push(false);
        atom.quaterion.push(UnitQuaternion::identity());
        atom.omega_x.push(0.0); atom.omega_y.push(0.0); atom.omega_z.push(0.0);
        atom.ang_mom_x.push(0.0); atom.ang_mom_y.push(0.0); atom.ang_mom_z.push(0.0);
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
        app.add_resource(mddem_core::CommResource(Box::new(mddem_core::SingleProcessComm::new())));
        app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
        app.add_update_system(bin_neighbor_list, ScheduleSet::Neighbor);
        app.organize_systems();
        app.setup();
        app.run();

        let n = app.get_resource_ref::<Neighbor>().unwrap();
        assert!(n.neighbor_list.len() >= 1, "bin neighbor list should find the close pair");
        let has_pair = n.neighbor_list.iter().any(|&(i, j)| (i == 0 && j == 1) || (i == 1 && j == 0));
        assert!(has_pair, "pair (0,1) should be in neighbor list");
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
        app.add_resource(mddem_core::CommResource(Box::new(mddem_core::SingleProcessComm::new())));
        app.add_update_system(sweep_and_prune_neighbor_list, ScheduleSet::Neighbor);
        app.organize_systems();
        app.run();

        let n = app.get_resource_ref::<Neighbor>().unwrap();
        assert_eq!(n.neighbor_list.len(), 1);
    }
}

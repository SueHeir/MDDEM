use std::process::exit;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

#[cfg(feature = "mpi_backend")]
use crate::ForceMPI;
use crate::{Atom, AtomDataRegistry, Config, Domain};

#[cfg(feature = "mpi_backend")]
use std::sync::Mutex;

#[cfg(feature = "mpi_backend")]
use mpi::collective::SystemOperation;
#[cfg(feature = "mpi_backend")]
use mpi::traits::{Communicator, CommunicatorCollectives, Destination, Source};

// ── CommConfig ──────────────────────────────────────────────────────────────

fn default_one_i32() -> i32 {
    1
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[comm]` — MPI processor grid configuration.
pub struct CommConfig {
    /// Number of MPI ranks in x dimension.
    #[serde(default = "default_one_i32")]
    pub processors_x: i32,
    /// Number of MPI ranks in y dimension.
    #[serde(default = "default_one_i32")]
    pub processors_y: i32,
    /// Number of MPI ranks in z dimension.
    #[serde(default = "default_one_i32")]
    pub processors_z: i32,
}

impl Default for CommConfig {
    fn default() -> Self {
        CommConfig {
            processors_x: 1,
            processors_y: 1,
            processors_z: 1,
        }
    }
}

// ── CommBackend trait ────────────────────────────────────────────────────────

/// Abstraction over MPI or single-process communication.
pub trait CommBackend: Send + Sync + 'static {
    fn rank(&self) -> i32;
    fn size(&self) -> i32;
    fn processor_decomposition(&self) -> Vector3<i32>;
    fn processor_position(&self) -> Vector3<i32>;
    fn set_processor_grid(&mut self, decomp: Vector3<i32>, position: Vector3<i32>);
    fn all_reduce_sum_f64(&self, local: f64) -> f64;
    fn all_reduce_min_f64(&self, local: f64) -> f64;
    fn barrier(&self);
}

/// Wraps a [`CommBackend`] implementation, used as `Res<CommResource>` in systems.
pub struct CommResource(pub Box<dyn CommBackend>);

impl std::ops::Deref for CommResource {
    type Target = dyn CommBackend;
    fn deref(&self) -> &(dyn CommBackend + 'static) {
        &*self.0
    }
}

impl std::ops::DerefMut for CommResource {
    fn deref_mut(&mut self) -> &mut (dyn CommBackend + 'static) {
        &mut *self.0
    }
}

// ── SingleProcessComm backend ────────────────────────────────────────────────

/// No-op communication backend for single-process simulations.
pub struct SingleProcessComm {
    processor_decomposition: Vector3<i32>,
    processor_position: Vector3<i32>,
}

impl Default for SingleProcessComm {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleProcessComm {
    pub fn new() -> Self {
        SingleProcessComm {
            processor_decomposition: Vector3::new(1, 1, 1),
            processor_position: Vector3::new(0, 0, 0),
        }
    }
}

impl CommBackend for SingleProcessComm {
    fn rank(&self) -> i32 {
        0
    }
    fn size(&self) -> i32 {
        1
    }
    fn processor_decomposition(&self) -> Vector3<i32> {
        self.processor_decomposition
    }
    fn processor_position(&self) -> Vector3<i32> {
        self.processor_position
    }

    fn set_processor_grid(&mut self, decomp: Vector3<i32>, position: Vector3<i32>) {
        self.processor_decomposition = decomp;
        self.processor_position = position;
    }

    fn all_reduce_sum_f64(&self, local: f64) -> f64 {
        local
    }
    fn all_reduce_min_f64(&self, local: f64) -> f64 {
        local
    }
    fn barrier(&self) {}
}

// ── SingleProcessCommPlugin ──────────────────────────────────────────────────

/// Single-process communication plugin with periodic ghost atom exchange.
pub struct SingleProcessCommPlugin;

impl Plugin for SingleProcessCommPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[comm]
# Number of MPI processors in each dimension
processors_x = 1
processors_y = 1
processors_z = 1"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<CommConfig>(app, "comm");

        app.add_resource(CommResource(Box::new(SingleProcessComm::new())));

        app.add_setup_system(comm_read_input, ScheduleSetupSet::PreSetup)
            .add_update_system(single_process_borders.label("single_process_borders"), ScheduleSet::PreNeighbor)
            .add_update_system(single_process_reverse_force, ScheduleSet::PostForce);

        app.add_cleanup(finalize_mpi);
    }
}

fn single_process_borders(
    comm: Res<CommResource>,
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
) {
    let local_count = atoms.len() as f64;
    let global_count = comm.all_reduce_sum_f64(local_count);
    atoms.natoms = global_count as u64;
    atoms.nlocal = atoms.len() as u32;
    atoms.nghost = 0;

    let mut send_buff: Vec<f64> = Vec::new();
    let mut scan_end = atoms.nlocal as usize;

    for dim in 0..3 {
        let dim_scan_end = scan_end;
        for swap in 0..2 {
            let periodic = match dim {
                0 => domain.is_periodic.x,
                1 => domain.is_periodic.y,
                2 => domain.is_periodic.z,
                _ => false,
            };
            if !periodic {
                continue;
            }

            let periodic_offset = if swap == 0 { 1.0 } else { -1.0 };
            send_buff.clear();

            let count = pack_border_atoms(
                &mut atoms,
                &registry,
                dim,
                swap,
                periodic_offset,
                &domain,
                &mut send_buff,
                dim_scan_end,
                domain.ghost_cutoff,
            );

            if count > 0 {
                let mut self_buf = send_buff.clone();
                self_buf.push(count as f64);
                unpack_ghost_atoms(&mut atoms, &registry, &self_buf, count as usize);
            }
        }
        scan_end = atoms.nlocal as usize + atoms.nghost as usize;
    }
}

fn single_process_reverse_force(mut atoms: ResMut<Atom>, comm: Res<CommResource>) {
    if comm.size() > 1 {
        return;
    }
    let nlocal = atoms.nlocal as usize;
    for i in (nlocal..atoms.len()).rev() {
        let origin = atoms.origin_index[i] as usize;
        atoms.force_x[origin] += atoms.force_x[i];
        atoms.force_y[origin] += atoms.force_y[i];
        atoms.force_z[origin] += atoms.force_z[i];
        atoms.torque_x[origin] += atoms.torque_x[i];
        atoms.torque_y[origin] += atoms.torque_y[i];
        atoms.torque_z[origin] += atoms.torque_z[i];
    }
}

// ── MPI backend ──────────────────────────────────────────────────────────────

#[cfg(feature = "mpi_backend")]
static MPI_UNIVERSE: Mutex<Option<mpi::environment::Universe>> = Mutex::new(None);

#[cfg(feature = "mpi_backend")]
fn get_mpi_world() -> mpi::topology::SimpleCommunicator {
    let mut guard = MPI_UNIVERSE.lock().unwrap();
    if guard.is_none() {
        *guard = Some(mpi::initialize().unwrap());
    }
    let universe = guard.as_ref().unwrap();
    universe.world()
}

/// Drop the MPI universe, calling MPI_Finalize. Must be called after all
/// `Comm` resources have been dropped (i.e. after the last `App` is done).
#[cfg(feature = "mpi_backend")]
pub fn finalize_mpi() {
    let mut guard = MPI_UNIVERSE.lock().unwrap();
    *guard = None;
}

#[cfg(not(feature = "mpi_backend"))]
pub fn finalize_mpi() {}

#[cfg(feature = "mpi_backend")]
pub struct MpiCommBackend {
    world: mpi::topology::SimpleCommunicator,
    rank: i32,
    size: i32,
    processor_decomposition: Vector3<i32>,
    processor_position: Vector3<i32>,
}

#[cfg(feature = "mpi_backend")]
unsafe impl Send for MpiCommBackend {}
#[cfg(feature = "mpi_backend")]
unsafe impl Sync for MpiCommBackend {}

#[cfg(feature = "mpi_backend")]
impl CommBackend for MpiCommBackend {
    fn rank(&self) -> i32 {
        self.rank
    }
    fn size(&self) -> i32 {
        self.size
    }
    fn processor_decomposition(&self) -> Vector3<i32> {
        self.processor_decomposition
    }
    fn processor_position(&self) -> Vector3<i32> {
        self.processor_position
    }

    fn set_processor_grid(&mut self, decomp: Vector3<i32>, position: Vector3<i32>) {
        self.processor_decomposition = decomp;
        self.processor_position = position;
    }

    fn all_reduce_sum_f64(&self, local: f64) -> f64 {
        let mut result = 0.0f64;
        self.world
            .all_reduce_into(&local, &mut result, SystemOperation::sum());
        result
    }

    fn all_reduce_min_f64(&self, local: f64) -> f64 {
        let mut result = 0.0f64;
        self.world
            .all_reduce_into(&local, &mut result, SystemOperation::min());
        result
    }

    fn barrier(&self) {
        self.world.barrier();
    }
}

#[cfg(feature = "mpi_backend")]
pub struct MpiCommInternal {
    pub world: mpi::topology::SimpleCommunicator,
    pub swap_directions: [Vector3<i32>; 2],
    pub periodic_swap: [Vector3<f64>; 2],
    pub send_amount: [Vector3<i32>; 2],
    pub receive_amount: [Vector3<i32>; 2],
}

#[cfg(feature = "mpi_backend")]
unsafe impl Send for MpiCommInternal {}
#[cfg(feature = "mpi_backend")]
unsafe impl Sync for MpiCommInternal {}

// ── MPI Plugin ──────────────────────────────────────────────────────────────

#[cfg(feature = "mpi_backend")]
pub struct CommunicationPlugin;

#[cfg(feature = "mpi_backend")]
impl Plugin for CommunicationPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[comm]
# Number of MPI processors in each dimension
processors_x = 1
processors_y = 1
processors_z = 1"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<CommConfig>(app, "comm");

        let world1 = get_mpi_world();
        let world2 = get_mpi_world();
        let rank = world1.rank();
        let size = world1.size();

        app.add_resource(CommResource(Box::new(MpiCommBackend {
            world: world1,
            rank,
            size,
            processor_decomposition: Vector3::zeros(),
            processor_position: Vector3::zeros(),
        })));

        app.add_resource(MpiCommInternal {
            world: world2,
            swap_directions: [Vector3::new(-1, -1, -1), Vector3::new(-1, -1, -1)],
            periodic_swap: [Vector3::zeros(), Vector3::zeros()],
            send_amount: [Vector3::new(0, 0, 0), Vector3::new(0, 0, 0)],
            receive_amount: [Vector3::new(0, 0, 0), Vector3::new(0, 0, 0)],
        });

        app.add_setup_system(comm_read_input, ScheduleSetupSet::PreSetup)
            .add_setup_system(comm_setup, ScheduleSetupSet::PostSetup)
            .add_update_system(exchange, ScheduleSet::Exchange)
            .add_update_system(borders.label("borders"), ScheduleSet::PreNeighbor)
            .add_update_system(reverse_send_force, ScheduleSet::PostForce);

        app.add_cleanup(finalize_mpi);
    }
}

// ── Shared systems ───────────────────────────────────────────────────────────

pub fn comm_read_input(config: Res<CommConfig>, mut comm: ResMut<CommResource>) {
    if comm.rank() == 0 {
        println!(
            "Comm: processors {} {} {}",
            config.processors_x, config.processors_y, config.processors_z
        );
    }

    let decomp = Vector3::new(
        config.processors_x,
        config.processors_y,
        config.processors_z,
    );
    let mul = config.processors_x * config.processors_y * config.processors_z;
    if mul != comm.size() {
        if comm.rank() == 0 {
            println!(
                "processors {} {} {} with {} processors does not match {}",
                config.processors_x,
                config.processors_y,
                config.processors_z,
                mul,
                comm.size()
            );
        }
        exit(1);
    }

    let rank = comm.rank();
    let pz = config.processors_z;
    let py = config.processors_y;
    let position = Vector3::new(
        rank / (py * pz),
        (rank / pz) % py,
        rank % pz,
    );

    comm.set_processor_grid(decomp, position);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn pack_border_atoms(
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    dim: usize,
    swap: usize,
    periodic_offset: f64,
    domain: &Domain,
    send_buff: &mut Vec<f64>,
    scan_end: usize,
    ghost_cutoff: f64,
) -> i32 {
    let mut count = 0i32;
    // Use ghost_cutoff if set (> 0), otherwise fall back to per-atom skin * 4.0 (DEM default)
    for i in 0..scan_end {
        let pos_dim = atoms.pos_component(i, dim);
        let cut = if ghost_cutoff > 0.0 { ghost_cutoff } else { atoms.skin[i] * 4.0 };
        let in_skin = if swap == 0 {
            pos_dim < domain.sub_domain_low[dim] + cut
        } else {
            pos_dim >= domain.sub_domain_high[dim] - cut
        };
        if in_skin {
            let mut change_pos = Vector3::zeros();
            change_pos[dim] = periodic_offset * domain.size[dim];
            atoms.pack_border(i, change_pos, send_buff);
            registry.pack_all(i, send_buff);
            count += 1;
        }
    }
    count
}

fn unpack_ghost_atoms(atoms: &mut Atom, registry: &AtomDataRegistry, buf: &[f64], count: usize) {
    atoms.reserve(count);
    let data = &buf[..buf.len() - 1];
    let mut pos = 0;
    for _ in 0..count {
        pos += atoms.unpack_atom(&data[pos..], true);
        pos += registry.unpack_all(&data[pos..]);
        atoms.nghost += 1;
    }
}

// ── MPI-only systems ─────────────────────────────────────────────────────────

#[cfg(feature = "mpi_backend")]
fn pos_to_rank(pos: Vector3<i32>, decomp: Vector3<i32>) -> i32 {
    pos.x * decomp.y * decomp.z + pos.y * decomp.z + pos.z
}

#[cfg(feature = "mpi_backend")]
pub fn comm_setup(comm: Res<CommResource>, mut mpi: ResMut<MpiCommInternal>, domain: Res<Domain>) {
    let decomp = comm.processor_decomposition();
    let pos = comm.processor_position();
    let periodic = [domain.is_periodic.x, domain.is_periodic.y, domain.is_periodic.z];
    let decomp_arr = [decomp.x, decomp.y, decomp.z];
    let pos_arr = [pos.x, pos.y, pos.z];

    for dim in 0..3 {
        // Forward neighbor (+1 in this dimension)
        if pos_arr[dim] + 1 < decomp_arr[dim] {
            let mut neighbor_pos = pos;
            neighbor_pos[dim] += 1;
            mpi.swap_directions[1][dim] = pos_to_rank(neighbor_pos, decomp);
        } else if periodic[dim] {
            let mut neighbor_pos = pos;
            neighbor_pos[dim] = 0;
            mpi.swap_directions[1][dim] = pos_to_rank(neighbor_pos, decomp);
            mpi.periodic_swap[1][dim] = -1.0;
        }

        // Backward neighbor (-1 in this dimension)
        if pos_arr[dim] - 1 >= 0 {
            let mut neighbor_pos = pos;
            neighbor_pos[dim] -= 1;
            mpi.swap_directions[0][dim] = pos_to_rank(neighbor_pos, decomp);
        } else if periodic[dim] {
            let mut neighbor_pos = pos;
            neighbor_pos[dim] = decomp_arr[dim] - 1;
            mpi.swap_directions[0][dim] = pos_to_rank(neighbor_pos, decomp);
            mpi.periodic_swap[0][dim] = 1.0;
        }
    }
}

// ── Exchange ──────────────────────────────────────────────────────────────────

#[cfg(feature = "mpi_backend")]
pub fn exchange(
    comm: Res<CommResource>,
    mpi: Res<MpiCommInternal>,
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
) {
    let size = comm.size();
    let rank = comm.rank();
    let proc_decomp = comm.processor_decomposition();
    let mut atoms_buff: Vec<Vec<f64>> = (0..size).map(|_| Vec::new()).collect();
    let mut counts = vec![0.0f64; size as usize];

    for i in (0..atoms.len()).rev() {
        let xi = (atoms.pos_x[i] / domain.sub_length.x).floor() as i32;
        let yi = (atoms.pos_y[i] / domain.sub_length.y).floor() as i32;
        let zi = (atoms.pos_z[i] / domain.sub_length.z).floor() as i32;
        let to_processor = xi * proc_decomp.z * proc_decomp.y + yi * proc_decomp.z + zi;
        if to_processor != rank {
            counts[to_processor as usize] += 1.0;
            atoms.pack_exchange(i, &mut atoms_buff[to_processor as usize]);
            registry.pack_all(i, &mut atoms_buff[to_processor as usize]);
            atoms.swap_remove(i);
            registry.swap_remove_all(i);
        }
    }

    for (buf, count) in atoms_buff.iter_mut().zip(counts) {
        buf.push(count);
    }

    for p in 0..size {
        if p == rank {
            for _rec in 0..size - 1 {
                let (msg, _status) = mpi.world.any_process().receive_vec::<f64>();
                let msg_count = msg[msg.len() - 1] as usize;
                let data = &msg[..msg.len() - 1];
                let mut pos = 0;
                for _ in 0..msg_count {
                    pos += atoms.unpack_atom(&data[pos..], false);
                    pos += registry.unpack_all(&data[pos..]);
                }
            }
        } else {
            mpi.world.process_at_rank(p).send(&atoms_buff[p as usize]);
        }
    }
    mpi.world.barrier();
}

// ── Borders ───────────────────────────────────────────────────────────────────

#[cfg(feature = "mpi_backend")]
pub fn borders(
    comm: Res<CommResource>,
    mut mpi: ResMut<MpiCommInternal>,
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
) {
    let local_count = atoms.len() as f64;
    let global_count = comm.all_reduce_sum_f64(local_count);
    atoms.natoms = global_count as u64;
    atoms.nlocal = atoms.len() as u32;
    atoms.nghost = 0;
    mpi.world.barrier();

    let rank = comm.rank();
    let mut send_buff: Vec<f64> = Vec::new();
    let mut scan_end = atoms.nlocal as usize;

    for dim in 0..3 {
        let dim_scan_end = scan_end;
        for swap in 0..2 {
            let to_proc = mpi.swap_directions[swap][dim];
            let from_proc = mpi.swap_directions[(swap + 1) % 2][dim];
            let periodic_offset = mpi.periodic_swap[swap][dim];
            mpi.send_amount[swap][dim] = 0;
            mpi.receive_amount[swap][dim] = 0;
            send_buff.clear();

            if to_proc == from_proc && to_proc != rank {
                if to_proc > rank {
                    if to_proc != -1 {
                        let count = pack_border_atoms(
                            &mut atoms,
                            &registry,
                            dim,
                            swap,
                            periodic_offset,
                            &domain,
                            &mut send_buff,
                            dim_scan_end,
                            domain.ghost_cutoff,
                        );
                        mpi.send_amount[swap][dim] = count;
                        send_buff.push(count as f64);
                        mpi.world.process_at_rank(to_proc).send(&send_buff);
                        let (msg, _status) =
                            mpi.world.process_at_rank(from_proc).receive_vec::<f64>();
                        let recv_count = msg[msg.len() - 1] as usize;
                        mpi.receive_amount[swap][dim] = recv_count as i32;
                        unpack_ghost_atoms(&mut atoms, &registry, &msg, recv_count);
                    }
                } else {
                    let count = pack_border_atoms(
                        &mut atoms,
                        &registry,
                        dim,
                        swap,
                        periodic_offset,
                        &domain,
                        &mut send_buff,
                        dim_scan_end,
                        domain.ghost_cutoff,
                    );
                    mpi.send_amount[swap][dim] = count;
                    if from_proc != -1 {
                        let (msg, _status) =
                            mpi.world.process_at_rank(to_proc).receive_vec::<f64>();
                        let recv_count = msg[msg.len() - 1] as usize;
                        mpi.receive_amount[swap][dim] = recv_count as i32;
                        unpack_ghost_atoms(&mut atoms, &registry, &msg, recv_count);
                        send_buff.push(count as f64);
                        mpi.world.process_at_rank(to_proc).send(&send_buff);
                    }
                }
            } else {
                if to_proc != -1 {
                    let count = pack_border_atoms(
                        &mut atoms,
                        &registry,
                        dim,
                        swap,
                        periodic_offset,
                        &domain,
                        &mut send_buff,
                        dim_scan_end,
                        domain.ghost_cutoff,
                    );
                    mpi.send_amount[swap][dim] = count;
                    if to_proc != rank {
                        send_buff.push(count as f64);
                        mpi.world.process_at_rank(to_proc).send(&send_buff);
                    } else {
                        mpi.receive_amount[swap][dim] = count;
                        let mut self_buf = send_buff.clone();
                        self_buf.push(count as f64);
                        unpack_ghost_atoms(&mut atoms, &registry, &self_buf, count as usize);
                    }
                }
                if from_proc != -1 && from_proc != rank {
                    let (msg, _status) = mpi.world.process_at_rank(from_proc).receive_vec::<f64>();
                    let recv_count = msg[msg.len() - 1] as usize;
                    mpi.receive_amount[swap][dim] = recv_count as i32;
                    unpack_ghost_atoms(&mut atoms, &registry, &msg, recv_count);
                }
            }
            mpi.world.barrier();
        }
        scan_end = atoms.nlocal as usize + atoms.nghost as usize;
    }
    mpi.world.barrier();
}

// ── Reverse send force ────────────────────────────────────────────────────────

#[cfg(feature = "mpi_backend")]
pub fn reverse_send_force(
    comm: Res<CommResource>,
    mpi: Res<MpiCommInternal>,
    mut atoms: ResMut<Atom>,
) {
    let rank = comm.rank();
    let mut send_buff: Vec<ForceMPI> = Vec::new();
    let mut receive_position = atoms.len() as i32;

    for dim in (0..3).rev() {
        for swap in (0..2).rev() {
            let to_proc = mpi.swap_directions[swap][dim];
            let from_proc = mpi.swap_directions[(swap + 1) % 2][dim];
            send_buff.clear();

            if to_proc == from_proc && to_proc != rank {
                if from_proc < rank {
                    if from_proc != -1 {
                        for i in
                            (receive_position - mpi.receive_amount[swap][dim])..receive_position
                        {
                            send_buff.push(atoms.get_force_data(i as usize));
                        }
                        mpi.world.process_at_rank(from_proc).send(&send_buff);
                        let (msg, _status) =
                            mpi.world.process_at_rank(to_proc).receive_vec::<ForceMPI>();
                        for atom in msg {
                            atoms.apply_force_data(atom);
                        }
                    }
                } else {
                    for i in (receive_position - mpi.receive_amount[swap][dim])..receive_position {
                        send_buff.push(atoms.get_force_data(i as usize));
                    }
                    if from_proc != -1 {
                        let (msg, _status) =
                            mpi.world.process_at_rank(to_proc).receive_vec::<ForceMPI>();
                        for atom in msg {
                            atoms.apply_force_data(atom);
                        }
                        mpi.world.process_at_rank(from_proc).send(&send_buff);
                    }
                }
            } else {
                if from_proc != -1 {
                    for i in (receive_position - mpi.receive_amount[swap][dim])..receive_position {
                        send_buff.push(atoms.get_force_data(i as usize));
                    }
                    if from_proc != rank {
                        mpi.world.process_at_rank(from_proc).send(&send_buff);
                    } else {
                        let msg = send_buff.clone();
                        for atom in msg {
                            atoms.apply_force_data(atom);
                        }
                    }
                }
                if to_proc != -1 && to_proc != rank {
                    let (msg, _status) =
                        mpi.world.process_at_rank(to_proc).receive_vec::<ForceMPI>();
                    for atom in msg {
                        atoms.apply_force_data(atom);
                    }
                }
            }
            receive_position -= mpi.receive_amount[swap][dim];
            mpi.world.barrier();
        }
    }
    mpi.world.barrier();
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_process_comm_rank_and_size() {
        let comm = SingleProcessComm::new();
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.size(), 1);
    }

    #[test]
    fn single_process_comm_reduce_identity() {
        let comm = SingleProcessComm::new();
        assert_eq!(comm.all_reduce_sum_f64(42.0), 42.0);
        assert_eq!(comm.all_reduce_min_f64(7.5), 7.5);
    }

    #[test]
    fn single_process_comm_set_grid() {
        let mut comm = SingleProcessComm::new();
        let decomp = Vector3::new(1, 1, 1);
        let pos = Vector3::new(0, 0, 0);
        comm.set_processor_grid(decomp, pos);
        assert_eq!(comm.processor_decomposition(), decomp);
        assert_eq!(comm.processor_position(), pos);
    }
}

//! MPI and single-process communication: ghost atoms, exchange, and force reduction.
//!
//! This module provides:
//! - [`CommBackend`] trait abstracting MPI or single-process communication
//! - [`SingleProcessComm`]: no-op backend for serial runs
//! - [`MpiCommBackend`]: real MPI backend (behind the `mpi_backend` feature)
//! - [`CommunicationPlugin`]: sets up the processor grid, ghost border
//!   communication, and reverse force reduction
//!
//! # Communication phases (per timestep)
//!
//! 1. **Exchange** (MPI only): migrate atoms that left the sub-domain
//! 2. **Borders**: create ghost copies of near-boundary atoms on neighboring ranks
//! 3. **Forward comm**: lightweight position/velocity update for existing ghosts
//! 4. **Reverse comm**: accumulate forces from ghosts back to their owners

use std::process::exit;

use sim_app::prelude::*;
use sim_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{Atom, AtomDataRegistry, CommState, Config, Domain, Neighbor, ParticleSimScheduleSet, ScheduleSetupSet};

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
    fn processor_decomposition(&self) -> [i32; 3];
    fn processor_position(&self) -> [i32; 3];
    fn set_processor_grid(&mut self, decomp: [i32; 3], position: [i32; 3]);
    fn all_reduce_sum_f64(&self, local: f64) -> f64;
    fn all_reduce_min_f64(&self, local: f64) -> f64;
    fn barrier(&self);

    // Point-to-point communication for borders/exchange/reverse_send_force
    fn send_f64(&self, dest: i32, buf: &[f64]);
    fn recv_f64(&self, source: i32) -> Vec<f64>;
    fn recv_f64_any(&self) -> Vec<f64>;
    // Deadlock-free sendrecv: send to dest while receiving from source
    fn sendrecv_f64(&self, dest: i32, send_buf: &[f64], source: i32) -> Vec<f64>;
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
    processor_decomposition: [i32; 3],
    processor_position: [i32; 3],
}

impl Default for SingleProcessComm {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleProcessComm {
    pub fn new() -> Self {
        SingleProcessComm {
            processor_decomposition: [1, 1, 1],
            processor_position: [0, 0, 0],
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
    fn processor_decomposition(&self) -> [i32; 3] {
        self.processor_decomposition
    }
    fn processor_position(&self) -> [i32; 3] {
        self.processor_position
    }

    fn set_processor_grid(&mut self, decomp: [i32; 3], position: [i32; 3]) {
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

    // Single-process always hits the to_proc == rank (local copy) branch,
    // so actual send/recv is never called.
    fn send_f64(&self, _dest: i32, _buf: &[f64]) {
        unreachable!("SingleProcessComm::send_f64 should never be called");
    }
    fn recv_f64(&self, _source: i32) -> Vec<f64> {
        unreachable!("SingleProcessComm::recv_f64 should never be called");
    }
    fn recv_f64_any(&self) -> Vec<f64> {
        unreachable!("SingleProcessComm::recv_f64_any should never be called");
    }
    fn sendrecv_f64(&self, _dest: i32, _send_buf: &[f64], _source: i32) -> Vec<f64> {
        unreachable!("SingleProcessComm::sendrecv_f64 should never be called");
    }
}

// ── CommBuffers ──────────────────────────────────────────────────────────────

/// Persistent communication buffers, reused across timesteps to avoid re-allocation.
pub struct CommBuffers {
    pub border_send_buff: Vec<f64>,
    /// Per-rank exchange buffers, reused across exchange() calls.
    pub exchange_buffs: Vec<Vec<f64>>,
}

impl Default for CommBuffers {
    fn default() -> Self {
        CommBuffers {
            border_send_buff: Vec::new(),
            exchange_buffs: Vec::new(),
        }
    }
}

// ── SwapData ─────────────────────────────────────────────────────────────────

/// Saved sendlist data for a single border swap, enabling lightweight forward_comm
/// position updates without full ghost rebuild.
pub struct SwapData {
    pub send_indices: Vec<usize>,   // atom indices packed in this swap
    pub recv_start: usize,          // first ghost index for received atoms
    pub recv_count: usize,          // number of ghosts received
    pub to_proc: i32,
    pub from_proc: i32,
    pub periodic_offset: [f64; 3],  // precomputed position shift
}

// ── CommTopology ─────────────────────────────────────────────────────────────

/// Swap directions and counts for ghost communication (replaces MpiCommInternal).
/// Works for both single-process and MPI.
pub struct CommTopology {
    pub swap_directions: [[i32; 3]; 2],
    pub periodic_swap: [[f64; 3]; 2],
    pub swap_data: Vec<SwapData>,   // sendlists for forward_comm
    pub maxneed: [i32; 3],          // number of swap layers per dimension (0 = not yet computed)
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
    processor_decomposition: [i32; 3],
    processor_position: [i32; 3],
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
    fn processor_decomposition(&self) -> [i32; 3] {
        self.processor_decomposition
    }
    fn processor_position(&self) -> [i32; 3] {
        self.processor_position
    }

    fn set_processor_grid(&mut self, decomp: [i32; 3], position: [i32; 3]) {
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

    fn send_f64(&self, dest: i32, buf: &[f64]) {
        self.world.process_at_rank(dest).send(buf);
    }

    fn recv_f64(&self, source: i32) -> Vec<f64> {
        let (msg, _status) = self.world.process_at_rank(source).receive_vec::<f64>();
        msg
    }

    fn recv_f64_any(&self) -> Vec<f64> {
        let (msg, _status) = self.world.any_process().receive_vec::<f64>();
        msg
    }

    fn sendrecv_f64(&self, dest: i32, send_buf: &[f64], source: i32) -> Vec<f64> {
        // Non-blocking send + blocking recv: deadlock-free for any dest/source combination
        let world = &self.world;
        mpi::request::scope(|scope| {
            let sreq = world.process_at_rank(dest)
                .immediate_send(scope, send_buf);
            let (msg, _status) = world.process_at_rank(source).receive_vec::<f64>();
            sreq.wait();
            msg
        })
    }

}

/// Run condition: returns true when Newton's third law optimization is enabled.
fn newton_on(neighbor: Res<Neighbor>) -> bool {
    neighbor.newton
}

// ── Unified CommunicationPlugin ──────────────────────────────────────────────

/// Plugin that sets up communication infrastructure: processor grid, ghost
/// borders, exchange (MPI), and reverse force accumulation.
pub struct CommunicationPlugin;

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

        // Select backend based on feature flag
        #[cfg(feature = "mpi_backend")]
        {
            let world = get_mpi_world();
            let rank = world.rank();
            let size = world.size();
            app.add_resource(CommResource(Box::new(MpiCommBackend {
                world,
                rank,
                size,
                processor_decomposition: [0; 3],
                processor_position: [0; 3],
            })));
        }
        #[cfg(not(feature = "mpi_backend"))]
        {
            app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        }

        app.add_resource(CommBuffers::default());
        app.add_resource(CommTopology {
            swap_directions: [[-1, -1, -1], [-1, -1, -1]],
            periodic_swap: [[0.0; 3], [0.0; 3]],
            swap_data: Vec::new(),
            maxneed: [0, 0, 0],
        });

        app.add_setup_system(comm_read_input.run_if(first_stage_only()), ScheduleSetupSet::PreSetup)
            .add_setup_system(comm_setup.run_if(first_stage_only()), ScheduleSetupSet::PostSetup)
            .add_update_system(
                borders.run_if(in_state(CommState::FullRebuild)),
                ParticleSimScheduleSet::PreNeighbor,
            )
            .add_update_system(
                forward_comm_borders.run_if(in_state(CommState::CommunicateOnly)),
                ParticleSimScheduleSet::PreNeighbor,
            )
            .add_update_system(
                reverse_send_force.label("reverse_send_force").run_if(newton_on),
                ParticleSimScheduleSet::PostForce,
            );

        #[cfg(feature = "mpi_backend")]
        app.add_update_system(
            exchange.label("exchange").run_if(in_state(CommState::FullRebuild)),
            ParticleSimScheduleSet::Exchange,
        );

        app.add_cleanup(finalize_mpi);
    }
}

// ── Shared systems ───────────────────────────────────────────────────────────

/// Read `[comm]` config and set up the processor grid decomposition.
pub fn comm_read_input(config: Res<CommConfig>, mut comm: ResMut<CommResource>) {
    if comm.rank() == 0 {
        println!(
            "Comm: processors {} {} {}",
            config.processors_x, config.processors_y, config.processors_z
        );
    }

    let decomp = [
        config.processors_x,
        config.processors_y,
        config.processors_z,
    ];
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
    let position = [
        rank / (py * pz),
        (rank / pz) % py,
        rank % pz,
    ];

    comm.set_processor_grid(decomp, position);
}

/// Convert a 3D processor grid position to a linear MPI rank.
fn pos_to_rank(pos: [i32; 3], decomp: [i32; 3]) -> i32 {
    pos[0] * decomp[1] * decomp[2] + pos[1] * decomp[2] + pos[2]
}

/// Compute neighbor ranks and periodic swap offsets for each dimension.
pub fn comm_setup(comm: Res<CommResource>, mut topo: ResMut<CommTopology>, domain: Res<Domain>) {
    let decomp = comm.processor_decomposition();
    let pos = comm.processor_position();
    let periodic = domain.periodic_flags();

    for dim in 0..3 {
        // Forward neighbor (+1 in this dimension)
        if pos[dim] + 1 < decomp[dim] {
            let mut neighbor_pos = pos;
            neighbor_pos[dim] += 1;
            topo.swap_directions[1][dim] = pos_to_rank(neighbor_pos, decomp);
        } else if periodic[dim] {
            let mut neighbor_pos = pos;
            neighbor_pos[dim] = 0;
            topo.swap_directions[1][dim] = pos_to_rank(neighbor_pos, decomp);
            topo.periodic_swap[1][dim] = -1.0;
        }

        // Backward neighbor (-1 in this dimension)
        if pos[dim] >= 1 {
            let mut neighbor_pos = pos;
            neighbor_pos[dim] -= 1;
            topo.swap_directions[0][dim] = pos_to_rank(neighbor_pos, decomp);
        } else if periodic[dim] {
            let mut neighbor_pos = pos;
            neighbor_pos[dim] = decomp[dim] - 1;
            topo.swap_directions[0][dim] = pos_to_rank(neighbor_pos, decomp);
            topo.periodic_swap[0][dim] = 1.0;
        }
    }
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
) -> (i32, Vec<usize>) {
    let mut count = 0i32;
    let mut packed_indices = Vec::new();
    // Use ghost_cutoff if set (> 0), otherwise fall back to per-atom skin * 4.0 (DEM default)
    for i in 0..scan_end {
        let pos_dim = atoms.pos_component(i, dim);
        let cut = if ghost_cutoff > 0.0 { ghost_cutoff } else { atoms.cutoff_radius[i] * 4.0 };
        let in_skin = if swap == 0 {
            pos_dim < domain.sub_domain_low[dim] + cut
        } else {
            pos_dim >= domain.sub_domain_high[dim] - cut
        };
        if in_skin {
            let mut change_pos = [0.0; 3];
            change_pos[dim] = periodic_offset * domain.size[dim];
            atoms.pack_border(i, change_pos, send_buff);
            registry.pack_all(i, send_buff);
            packed_indices.push(i);
            count += 1;
        }
    }
    (count, packed_indices)
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

// ── Forward comm (lightweight ghost position update) ─────────────────────────

/// Per-atom stride in forward_comm: pos(3) + vel(3) = 6 f64s (base fields only).
const FORWARD_PACK_SIZE: usize = 6;

/// For self-sends (single-process periodic), compute the periodic offset for
/// an atom's ghost. Uses the stored swap direction to determine which periodic
/// image to create.
///
/// The stored_offset sign indicates the swap direction:
///   +size → ghost should be above (atom was near low boundary, swap=0)
///   -size → ghost should be below (atom was near high boundary, swap=1)
///
/// After PBC wrapping, the atom may have moved to the other side of the box.
/// We use the stored direction sign and the atom's current position to ensure
/// the ghost always ends up on the opposite side from the atom.
#[inline]
fn compute_per_atom_offset(
    pos: &[f64; 3],
    stored_offset: &[f64; 3],
    boundaries_low: &[f64; 3],
    boundaries_high: &[f64; 3],
    size: &[f64; 3],
) -> [f64; 3] {
    let mut offset = [0.0; 3];
    for dim in 0..3 {
        if stored_offset[dim] != 0.0 {
            // Use the stored direction from the original swap.
            // The swap direction tells us which periodic image to create:
            //   swap=0 (stored > 0): ghost should be at pos + size
            //   swap=1 (stored < 0): ghost should be at pos - size
            //
            // After PBC wrapping, the atom may now be on the opposite side.
            // If the atom crossed the boundary, the stored direction may now
            // point the ghost to the same side as the atom. Detect this by
            // checking if the resulting ghost would land inside the domain
            // (meaning the atom moved and we need to flip the sign).
            let sign = stored_offset[dim].signum();
            let ghost_pos = pos[dim] + sign * size[dim];

            // If the ghost would be well inside the domain, the atom has PBC-wrapped
            // and the original direction is stale — flip it.
            let lo = boundaries_low[dim];
            let hi = boundaries_high[dim];
            if ghost_pos > lo && ghost_pos < hi {
                offset[dim] = -sign * size[dim];
            } else {
                offset[dim] = sign * size[dim];
            }
        }
    }
    offset
}

fn unpack_forward(msg: &[f64], atoms: &mut Atom, registry: &AtomDataRegistry, recv_start: usize, recv_count: usize, extra_per_atom: usize) {
    let stride = FORWARD_PACK_SIZE + extra_per_atom;
    for k in 0..recv_count {
        let base = k * stride;
        atoms.pos[recv_start + k] = [msg[base], msg[base + 1], msg[base + 2]];
        atoms.vel[recv_start + k] = [msg[base + 3], msg[base + 4], msg[base + 5]];
        if extra_per_atom > 0 {
            registry.unpack_forward_all(recv_start + k, &msg[base + FORWARD_PACK_SIZE..]);
        }
    }
}

fn forward_comm(
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    topo: &CommTopology,
    comm: &dyn CommBackend,
    domain: &Domain,
    buf: &mut Vec<f64>,
) {
    let rank = comm.rank();
    let extra = registry.forward_comm_size();
    for swap in &topo.swap_data {
        buf.clear();

        // For self-sends (single-process periodic), recompute the periodic offset
        // per atom based on current position. After PBC wrapping, an atom originally
        // near one boundary may now be near the other, so the stored offset is stale.
        // Recomputing ensures the ghost ends up at the correct periodic image.
        let is_self_send = swap.to_proc == rank;

        // Pack positions and velocities (6 f64s per atom) + registry forward fields
        for &idx in &swap.send_indices {
            let offset = if is_self_send {
                compute_per_atom_offset(
                    &atoms.pos[idx],
                    &swap.periodic_offset,
                    &domain.boundaries_low,
                    &domain.boundaries_high,
                    &domain.size,
                )
            } else {
                swap.periodic_offset
            };
            buf.push(atoms.pos[idx][0] + offset[0]);
            buf.push(atoms.pos[idx][1] + offset[1]);
            buf.push(atoms.pos[idx][2] + offset[2]);
            buf.push(atoms.vel[idx][0]);
            buf.push(atoms.vel[idx][1]);
            buf.push(atoms.vel[idx][2]);
            registry.pack_forward_all(idx, buf);
        }

        if swap.to_proc == rank {
            // Self-send: copy directly into ghost data
            unpack_forward(buf, atoms, registry, swap.recv_start, swap.recv_count, extra);
        } else {
            // MPI: sendrecv is deadlock-free regardless of symmetric/asymmetric
            if swap.to_proc != -1 && swap.from_proc != -1 {
                let msg = comm.sendrecv_f64(swap.to_proc, buf, swap.from_proc);
                unpack_forward(&msg, atoms, registry, swap.recv_start, swap.recv_count, extra);
            } else if swap.to_proc != -1 {
                comm.send_f64(swap.to_proc, buf);
            } else if swap.from_proc != -1 {
                let msg = comm.recv_f64(swap.from_proc);
                unpack_forward(&msg, atoms, registry, swap.recv_start, swap.recv_count, extra);
            }
        }
    }
}

// ── Unified borders ──────────────────────────────────────────────────────────

/// Lightweight ghost position update: forward-comm only (no full rebuild).
///
/// Gated by `run_if(in_state(CommState::CommunicateOnly))`.
/// Recomputes per-atom periodic offsets for self-sends, so ghost
/// positions stay correct even when atoms cross PBC boundaries between rebuilds.
pub fn forward_comm_borders(
    comm: Res<CommResource>,
    topo: Res<CommTopology>,
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
    mut buffers: ResMut<CommBuffers>,
) {
    let mut send_buff = std::mem::take(&mut buffers.border_send_buff);
    forward_comm(&mut atoms, &registry, &topo, &**comm, &domain, &mut send_buff);
    buffers.border_send_buff = send_buff;
}

/// Build ghost atoms by scanning near-boundary locals and sending them to neighbors.
///
/// Gated by `run_if(in_state(CommState::FullRebuild))`.
pub fn borders(
    comm: Res<CommResource>,
    mut topo: ResMut<CommTopology>,
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
    mut buffers: ResMut<CommBuffers>,
) {
    let mut send_buff = std::mem::take(&mut buffers.border_send_buff);

    // Full ghost rebuild: remove old ghosts first
    if atoms.nghost > 0 {
        atoms.truncate_to_nlocal();
        registry.truncate_all(atoms.nlocal as usize);
        atoms.nghost = 0;
    }

    let local_count = atoms.len() as f64;
    let global_count = comm.all_reduce_sum_f64(local_count);
    atoms.natoms = global_count as u64;
    atoms.nlocal = atoms.len() as u32;
    atoms.nghost = 0;

    // Compute maxneed on first full rebuild (ghost_cutoff is set by neighbor_setup)
    if topo.maxneed == [0, 0, 0] {
        let decomp = comm.processor_decomposition();
        for (dim, (&dec, &sub_len)) in decomp.iter().zip(domain.sub_length.iter()).enumerate() {
            if dec == 1 {
                topo.maxneed[dim] = 1;
            } else {
                topo.maxneed[dim] = (domain.ghost_cutoff / sub_len).ceil().max(1.0) as i32;
            }
        }
        if comm.rank() == 0 {
            for dim in 0..3 {
                if topo.maxneed[dim] > 1 {
                    let dim_name = ["x", "y", "z"][dim];
                    println!("Comm: multi-hop ghost communication in dim {}: need={}", dim_name, topo.maxneed[dim]);
                }
            }
        }
    }

    // Clear sendlists for fresh recording
    topo.swap_data.clear();

    let rank = comm.rank();
    let mut scan_end = atoms.nlocal as usize;

    for dim in 0..3 {
        for _need in 0..topo.maxneed[dim] {
            let dim_scan_end = scan_end;
            for swap in 0..2 {
                let to_proc = topo.swap_directions[swap][dim];
                let from_proc = topo.swap_directions[(swap + 1) % 2][dim];
                let periodic_offset = topo.periodic_swap[swap][dim];
                send_buff.clear();

                // Precompute periodic position shift for sendlist
                let mut pbc_shift = [0.0; 3];
                pbc_shift[dim] = periodic_offset * domain.size[dim];

                let (count, packed_indices) = pack_border_atoms(
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
                let swap_recv_start;
                let swap_recv_count;

                if to_proc == rank {
                    // Self-send (single-process periodic): unpack locally
                    send_buff.push(count as f64);
                    swap_recv_start = atoms.len();
                    swap_recv_count = count as usize;
                    unpack_ghost_atoms(&mut atoms, &registry, &send_buff, count as usize);
                } else if to_proc != -1 && from_proc != -1 {
                    // MPI: sendrecv is deadlock-free
                    send_buff.push(count as f64);
                    let msg = comm.sendrecv_f64(to_proc, &send_buff, from_proc);
                    let recv_count = msg[msg.len() - 1] as usize;
                    swap_recv_start = atoms.len();
                    swap_recv_count = recv_count;
                    unpack_ghost_atoms(&mut atoms, &registry, &msg, recv_count);
                } else if to_proc != -1 {
                    // Send only (non-periodic boundary, no neighbor to receive from)
                    send_buff.push(count as f64);
                    comm.send_f64(to_proc, &send_buff);
                    swap_recv_start = atoms.len();
                    swap_recv_count = 0;
                } else if from_proc != -1 {
                    // Receive only
                    let msg = comm.recv_f64(from_proc);
                    let recv_count = msg[msg.len() - 1] as usize;
                    swap_recv_start = atoms.len();
                    swap_recv_count = recv_count;
                    unpack_ghost_atoms(&mut atoms, &registry, &msg, recv_count);
                } else {
                    swap_recv_start = atoms.len();
                    swap_recv_count = 0;
                }

                topo.swap_data.push(SwapData {
                    send_indices: packed_indices,
                    recv_start: swap_recv_start,
                    recv_count: swap_recv_count,
                    to_proc,
                    from_proc,
                    periodic_offset: pbc_shift,
                });
            }
            scan_end = atoms.nlocal as usize + atoms.nghost as usize;
        }
    }
    buffers.border_send_buff = send_buff;
}

// ── Unified reverse send force ────────────────────────────────────────────────

/// Accumulate ghost forces back onto their owner atoms.
///
/// Iterates swap data in reverse order (mirroring `borders`), packing
/// force + registry reverse fields from each ghost and sending them to the
/// rank that owns the original atom.
pub fn reverse_send_force(
    comm: Res<CommResource>,
    topo: Res<CommTopology>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
) {
    let rank = comm.rank();
    let mut send_buff: Vec<f64> = Vec::new();

    // Compute per-atom stride once: 5 base (tag + origin_index + force×3) + registry reverse fields
    let per_atom = 5 + registry.reverse_comm_size();

    // Iterate swaps in reverse order (mirrors borders() forward order)
    for swap in topo.swap_data.iter().rev() {
        send_buff.clear();

        // Pack force + origin_index + tag (+ registry reverse fields) per ghost atom
        for i in swap.recv_start..(swap.recv_start + swap.recv_count) {
            debug_assert!(atoms.is_ghost[i], "reverse_send_force: atom {} is not ghost", i);
            send_buff.push(atoms.tag[i] as f64);
            send_buff.push(atoms.origin_index[i] as f64);
            send_buff.push(atoms.force[i][0]);
            send_buff.push(atoms.force[i][1]);
            send_buff.push(atoms.force[i][2]);
            registry.pack_reverse_all(i, &mut send_buff);
        }

        if swap.from_proc == rank {
            // Self-send: apply forces locally
            for k in 0..swap.recv_count {
                let base = k * per_atom;
                let origin = send_buff[base + 1] as usize;
                debug_assert_eq!(
                    send_buff[base] as u32, atoms.tag[origin],
                    "reverse_send_force: tag mismatch"
                );
                atoms.force[origin][0] += send_buff[base + 2];
                atoms.force[origin][1] += send_buff[base + 3];
                atoms.force[origin][2] += send_buff[base + 4];
                if per_atom > 5 {
                    registry.unpack_reverse_all(origin, &send_buff[base + 5..]);
                }
            }
        } else if swap.from_proc != -1 && swap.to_proc != -1 {
            let msg = comm.sendrecv_f64(swap.from_proc, &send_buff, swap.to_proc);
            let recv_count = msg.len() / per_atom;
            for k in 0..recv_count {
                let base = k * per_atom;
                let origin = msg[base + 1] as usize;
                debug_assert_eq!(
                    msg[base] as u32, atoms.tag[origin],
                    "reverse_send_force: tag mismatch"
                );
                atoms.force[origin][0] += msg[base + 2];
                atoms.force[origin][1] += msg[base + 3];
                atoms.force[origin][2] += msg[base + 4];
                if per_atom > 5 {
                    registry.unpack_reverse_all(origin, &msg[base + 5..]);
                }
            }
        } else if swap.from_proc != -1 {
            comm.send_f64(swap.from_proc, &send_buff);
        } else if swap.to_proc != -1 {
            let msg = comm.recv_f64(swap.to_proc);
            let recv_count = msg.len() / per_atom;
            for k in 0..recv_count {
                let base = k * per_atom;
                let origin = msg[base + 1] as usize;
                atoms.force[origin][0] += msg[base + 2];
                atoms.force[origin][1] += msg[base + 3];
                atoms.force[origin][2] += msg[base + 4];
                if per_atom > 5 {
                    registry.unpack_reverse_all(origin, &msg[base + 5..]);
                }
            }
        }
    }
}

// ── Exchange (MPI only) ──────────────────────────────────────────────────────

#[cfg(feature = "mpi_backend")]
pub fn exchange(
    comm: Res<CommResource>,
    topo: Res<CommTopology>,
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
    mut buffers: ResMut<CommBuffers>,
) {
    let decomp = comm.processor_decomposition();

    // Reuse persistent exchange buffers (only need 2: lo and hi per dimension)
    let mut atoms_buff = std::mem::take(&mut buffers.exchange_buffs);
    atoms_buff.resize_with(2, Vec::new);

    // Per-dimension exchange: atoms migrating in dim 0 are sent first,
    // received atoms may continue migrating in dims 1, 2.
    for dim in 0..3usize {
        if decomp[dim] == 1 {
            continue; // No exchange needed in this dimension (PBC handled elsewhere)
        }

        let lo_proc = topo.swap_directions[0][dim]; // neighbor in -dim direction
        let hi_proc = topo.swap_directions[1][dim]; // neighbor in +dim direction

        atoms_buff[0].clear(); // lo send buffer
        atoms_buff[1].clear(); // hi send buffer
        let mut lo_count = 0.0f64;
        let mut hi_count = 0.0f64;

        // Scan local atoms: pack those outside subdomain in this dimension
        for i in (0..atoms.len()).rev() {
            if atoms.pos[i][dim] < domain.sub_domain_low[dim] {
                lo_count += 1.0;
                atoms.pack_exchange(i, &mut atoms_buff[0]);
                registry.pack_all(i, &mut atoms_buff[0]);
                atoms.swap_remove(i);
                registry.swap_remove_all(i);
            } else if atoms.pos[i][dim] >= domain.sub_domain_high[dim] {
                hi_count += 1.0;
                atoms.pack_exchange(i, &mut atoms_buff[1]);
                registry.pack_all(i, &mut atoms_buff[1]);
                atoms.swap_remove(i);
                registry.swap_remove_all(i);
            }
        }

        atoms_buff[0].push(lo_count);
        atoms_buff[1].push(hi_count);

        // Send lo, receive from hi
        if lo_proc != -1 && hi_proc != -1 {
            let msg = comm.sendrecv_f64(lo_proc, &atoms_buff[0], hi_proc);
            let msg_count = msg[msg.len() - 1] as usize;
            let data = &msg[..msg.len() - 1];
            let mut pos = 0;
            for _ in 0..msg_count {
                pos += atoms.unpack_atom(&data[pos..], false);
                pos += registry.unpack_all(&data[pos..]);
            }
        } else if lo_proc != -1 {
            comm.send_f64(lo_proc, &atoms_buff[0]);
        } else if hi_proc != -1 {
            let msg = comm.recv_f64(hi_proc);
            let msg_count = msg[msg.len() - 1] as usize;
            let data = &msg[..msg.len() - 1];
            let mut pos = 0;
            for _ in 0..msg_count {
                pos += atoms.unpack_atom(&data[pos..], false);
                pos += registry.unpack_all(&data[pos..]);
            }
        }

        // Send hi, receive from lo
        if hi_proc != -1 && lo_proc != -1 {
            let msg = comm.sendrecv_f64(hi_proc, &atoms_buff[1], lo_proc);
            let msg_count = msg[msg.len() - 1] as usize;
            let data = &msg[..msg.len() - 1];
            let mut pos = 0;
            for _ in 0..msg_count {
                pos += atoms.unpack_atom(&data[pos..], false);
                pos += registry.unpack_all(&data[pos..]);
            }
        } else if hi_proc != -1 {
            comm.send_f64(hi_proc, &atoms_buff[1]);
        } else if lo_proc != -1 {
            let msg = comm.recv_f64(lo_proc);
            let msg_count = msg[msg.len() - 1] as usize;
            let data = &msg[..msg.len() - 1];
            let mut pos = 0;
            for _ in 0..msg_count {
                pos += atoms.unpack_atom(&data[pos..], false);
                pos += registry.unpack_all(&data[pos..]);
            }
        }
    }

    buffers.exchange_buffs = atoms_buff;
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
        let decomp = [1, 1, 1];
        let pos = [0, 0, 0];
        comm.set_processor_grid(decomp, pos);
        assert_eq!(comm.processor_decomposition(), decomp);
        assert_eq!(comm.processor_position(), pos);
    }

    #[test]
    fn pos_to_rank_basic() {
        let decomp = [2, 2, 1];
        assert_eq!(pos_to_rank([0, 0, 0], decomp), 0);
        assert_eq!(pos_to_rank([0, 1, 0], decomp), 1);
        assert_eq!(pos_to_rank([1, 0, 0], decomp), 2);
        assert_eq!(pos_to_rank([1, 1, 0], decomp), 3);
    }

    #[test]
    fn pos_to_rank_3d() {
        let decomp = [2, 2, 2];
        assert_eq!(pos_to_rank([0, 0, 0], decomp), 0);
        assert_eq!(pos_to_rank([0, 0, 1], decomp), 1);
        assert_eq!(pos_to_rank([0, 1, 0], decomp), 2);
        assert_eq!(pos_to_rank([1, 0, 0], decomp), 4);
        assert_eq!(pos_to_rank([1, 1, 1], decomp), 7);
    }

    #[test]
    fn compute_per_atom_offset_follows_stored_direction() {
        let boundaries_low = [0.0, 0.0, 0.0];
        let boundaries_high = [10.0, 10.0, 10.0];
        let size = [10.0, 10.0, 10.0];

        // stored > 0 means swap=0 (low boundary ghost, shift +size)
        // stored < 0 means swap=1 (high boundary ghost, shift -size)
        let stored = [10.0, 0.0, -10.0];

        // Atom in lower half: stored direction produces ghost outside box
        let pos = [2.0, 5.0, 3.0];
        let offset = compute_per_atom_offset(&pos, &stored, &boundaries_low, &boundaries_high, &size);
        assert_eq!(offset[0], 10.0);   // stored +, ghost at 12 (outside) → keep +size
        assert_eq!(offset[1], 0.0);    // no periodic
        assert_eq!(offset[2], -10.0);  // stored -, ghost at -7 (outside) → keep -size

        // Atom in upper half
        let pos = [8.0, 5.0, 8.0];
        let offset = compute_per_atom_offset(&pos, &stored, &boundaries_low, &boundaries_high, &size);
        assert_eq!(offset[0], 10.0);   // stored +, ghost at 18 (outside) → keep +size
        assert_eq!(offset[1], 0.0);
        assert_eq!(offset[2], -10.0);  // stored -, ghost at -2 (outside) → keep -size
    }

    #[test]
    fn compute_per_atom_offset_pbc_wrapped_atom() {
        let boundaries_low = [0.0, 0.0, 0.0];
        let boundaries_high = [10.0, 10.0, 10.0];
        let size = [10.0, 10.0, 10.0];

        // Atom was originally near high boundary (z=9), stored=-10 (swap=1).
        // After PBC wrapping, atom is now at z=1. The ghost should flip to +size.
        let stored = [0.0, 0.0, -10.0];
        let pos = [5.0, 5.0, 1.0];
        let offset = compute_per_atom_offset(&pos, &stored, &boundaries_low, &boundaries_high, &size);
        // ghost at 1-10=-9 → outside → keep -size
        assert_eq!(offset[2], -10.0);

        // Atom was originally near low boundary (x=1), stored=+10 (swap=0).
        // After PBC wrapping, atom is now at x=9. Ghost at 9+10=19 → outside → keep +size
        let stored = [10.0, 0.0, 0.0];
        let pos = [9.0, 5.0, 5.0];
        let offset = compute_per_atom_offset(&pos, &stored, &boundaries_low, &boundaries_high, &size);
        assert_eq!(offset[0], 10.0);
    }

    #[test]
    fn compute_per_atom_offset_overlap_zone() {
        // Critical test: atom in the overlap zone (near midpoint) with
        // different stored directions must produce DIFFERENT ghost positions.
        let boundaries_low = [0.0, 0.0, 0.0];
        let boundaries_high = [0.006, 0.006, 0.1];
        let size = [0.006, 0.006, 0.1];

        let pos = [0.003, 0.003, 0.05]; // at midpoint of y

        // swap=0 direction: stored_offset positive
        let stored_swap0 = [0.0, 0.006, 0.0];
        let offset0 = compute_per_atom_offset(&pos, &stored_swap0, &boundaries_low, &boundaries_high, &size);

        // swap=1 direction: stored_offset negative
        let stored_swap1 = [0.0, -0.006, 0.0];
        let offset1 = compute_per_atom_offset(&pos, &stored_swap1, &boundaries_low, &boundaries_high, &size);

        // The two ghosts MUST be at different positions
        assert_ne!(offset0[1], offset1[1], "Ghosts from different swaps must have different offsets");
        // swap=0: ghost above → +0.006
        assert_eq!(offset0[1], 0.006);
        // swap=1: ghost below → -0.006
        assert_eq!(offset1[1], -0.006);
    }

    #[test]
    fn swap_data_records_correct_info() {
        let swap = SwapData {
            send_indices: vec![0, 3, 7],
            recv_start: 100,
            recv_count: 5,
            to_proc: 1,
            from_proc: 2,
            periodic_offset: [10.0, 0.0, 0.0],
        };
        assert_eq!(swap.send_indices.len(), 3);
        assert_eq!(swap.recv_start, 100);
        assert_eq!(swap.recv_count, 5);
        assert_eq!(swap.to_proc, 1);
        assert_eq!(swap.from_proc, 2);
    }
}

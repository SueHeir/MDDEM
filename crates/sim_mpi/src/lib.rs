//! Pure MPI abstraction layer.
//!
//! Provides [`CommBackend`] as the communication interface, [`CommResource`] as a
//! resource wrapper, and two backends:
//! - [`SingleProcessComm`]: no-op backend for serial runs
//! - [`MpiCommBackend`]: real MPI backend (behind the `mpi_backend` feature)

use std::ops::{Deref, DerefMut};

#[cfg(feature = "mpi_backend")]
use std::sync::Mutex;

#[cfg(feature = "mpi_backend")]
use mpi::collective::SystemOperation;
#[cfg(feature = "mpi_backend")]
use mpi::traits::{Communicator, CommunicatorCollectives, Destination, Source};

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

// ── CommResource ─────────────────────────────────────────────────────────────

/// Wraps a [`CommBackend`] implementation, used as `Res<CommResource>` in systems.
pub struct CommResource(pub Box<dyn CommBackend>);

impl Deref for CommResource {
    type Target = dyn CommBackend;
    fn deref(&self) -> &(dyn CommBackend + 'static) {
        &*self.0
    }
}

impl DerefMut for CommResource {
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

// ── MPI backend ──────────────────────────────────────────────────────────────

#[cfg(feature = "mpi_backend")]
static MPI_UNIVERSE: Mutex<Option<mpi::environment::Universe>> = Mutex::new(None);

#[cfg(feature = "mpi_backend")]
pub fn get_mpi_world() -> mpi::topology::SimpleCommunicator {
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
impl MpiCommBackend {
    pub fn new(world: mpi::topology::SimpleCommunicator) -> Self {
        let rank = world.rank();
        let size = world.size();
        MpiCommBackend {
            world,
            rank,
            size,
            processor_decomposition: [0; 3],
            processor_position: [0; 3],
        }
    }
}

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
}

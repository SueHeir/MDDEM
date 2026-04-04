//! Per-atom data storage and the [`AtomData`] extension trait.
//!
//! This module provides:
//! - [`Atom`]: struct-of-arrays storage for core per-atom fields (position,
//!   velocity, force, mass, etc.)
//! - [`AtomData`]: trait for plugin-specific per-atom extensions (e.g. radius,
//!   angular velocity) with pack/unpack support for MPI communication
//! - [`AtomDataRegistry`]: dynamic registry that manages all [`AtomData`]
//!   extensions, keyed by `TypeId`
//! - [`AtomPlugin`]: registers the [`Atom`] and [`AtomDataRegistry`] resources
//!   and the per-step force zeroing system

use std::{
    any::{Any, TypeId},
    cell::{Ref, RefCell, RefMut},
};

use sim_app::prelude::*;
use sim_scheduler::prelude::*;
use crate::{CommState, ParticleSimScheduleSet};

/// Number of `f64`s packed/unpacked for one atom's base fields
/// (tag, origin_index, cutoff_radius, atom_type, pos×3, vel×3, force×3, mass, image×3).
pub const ATOM_PACK_SIZE: usize = 17;

// ── AtomData trait ───────────────────────────────────────────────────────────

/// Register an [`AtomData`] extension with the [`AtomDataRegistry`] stored in `app`.
///
/// Panics if `AtomPlugin` has not been added (i.e. `AtomDataRegistry` is missing).
///
/// # Example
/// ```ignore
/// register_atom_data!(app, DemAtom::new());
/// ```
#[macro_export]
macro_rules! register_atom_data {
    ($app:expr, $value:expr) => {
        if let Some(cell) = $app.get_mut_resource(::std::any::TypeId::of::<$crate::AtomDataRegistry>()) {
            let mut binder = cell.borrow_mut();
            binder.downcast_mut::<$crate::AtomDataRegistry>()
                .expect("Failed to downcast AtomDataRegistry — this is a bug in MDDEM")
                .register($value);
        } else {
            panic!("AtomDataRegistry not found — AtomPlugin must be added first");
        }
    };
}

/// Per-atom extension data (e.g. radius, density for DEM).
///
/// Implement this trait to attach custom per-atom fields to the simulation.
/// The registry handles pack/unpack for MPI ghost and exchange communication
/// automatically. Use the `#[derive(AtomData)]` macro from `mddem_derive` for
/// simple struct-of-Vec extensions; implement manually only for complex layouts
/// (like [`BondStore`](crate::BondStore)).
pub trait AtomData: Any {
    /// Upcast to `&dyn Any` for downcasting in the registry.
    fn as_any(&self) -> &dyn Any;
    /// Upcast to `&mut dyn Any` for mutable downcasting in the registry.
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Shrink all per-atom Vecs to length `n`, discarding trailing entries.
    fn truncate(&mut self, n: usize);
    /// Remove atom at index `i` by swapping with the last element.
    fn swap_remove(&mut self, i: usize);
    /// Serialize all fields for atom `i` into `buf` (exchange/border comm).
    fn pack(&self, i: usize, buf: &mut Vec<f64>);
    /// Deserialize one atom's fields from `buf`, appending to internal Vecs.
    /// Returns the number of `f64`s consumed.
    fn unpack(&mut self, buf: &[f64]) -> usize;
    /// Reorder atoms according to `perm[0..n]` (used by spatial sorting).
    fn apply_permutation(&mut self, perm: &[usize], n: usize);

    /// Pack forward-comm fields (e.g. omega for DEM) into buf.
    fn pack_forward(&self, _i: usize, _buf: &mut Vec<f64>) {}
    /// Unpack forward-comm fields; returns number of f64s consumed.
    fn unpack_forward(&mut self, _i: usize, _buf: &[f64]) -> usize { 0 }
    /// Pack reverse-comm fields (e.g. torque for DEM) into buf.
    fn pack_reverse(&self, _i: usize, _buf: &mut Vec<f64>) {}
    /// Unpack reverse-comm fields; returns number of f64s consumed.
    fn unpack_reverse(&mut self, _i: usize, _buf: &[f64]) -> usize { 0 }
    /// Zero per-step accumulators (e.g. torque) for atoms 0..n.
    fn zero(&mut self, _n: usize) {}
    /// Number of f64s per atom in forward comm.
    fn forward_comm_size(&self) -> usize { 0 }
    /// Number of f64s per atom in reverse comm.
    fn reverse_comm_size(&self) -> usize { 0 }
}

// ── AtomDataRegistry ─────────────────────────────────────────────────────────

/// Dynamic registry of [`AtomData`] extensions, keyed by `TypeId`.
pub struct AtomDataRegistry {
    stores: Vec<(TypeId, RefCell<Box<dyn AtomData>>)>,
}

impl Default for AtomDataRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomDataRegistry {
    pub fn new() -> Self {
        AtomDataRegistry {
            stores: Vec::new(),
        }
    }

    /// Register a new [`AtomData`] extension. Panics if the same type is registered twice.
    pub fn register<T: AtomData + 'static>(&mut self, data: T) {
        let id = TypeId::of::<T>();
        for (existing_id, _) in &self.stores {
            if *existing_id == id {
                panic!("AtomData type already registered");
            }
        }
        self.stores.push((id, RefCell::new(Box::new(data))));
    }

    /// Borrow a registered [`AtomData`] extension immutably, or `None` if not registered.
    pub fn get<T: AtomData + 'static>(&self) -> Option<Ref<'_, T>> {
        let id = TypeId::of::<T>();
        self.stores
            .iter()
            .find(|(tid, _)| *tid == id)
            .map(|(_, cell)| Ref::map(cell.borrow(), |b| {
                b.as_any().downcast_ref::<T>()
                    .expect("AtomData downcast failed — type mismatch (this is a bug in MDDEM)")
            }))
    }

    /// Borrow a registered [`AtomData`] extension mutably, or `None` if not registered.
    pub fn get_mut<T: AtomData + 'static>(&self) -> Option<RefMut<'_, T>> {
        let id = TypeId::of::<T>();
        self.stores
            .iter()
            .find(|(tid, _)| *tid == id)
            .map(|(_, cell)| {
                RefMut::map(cell.borrow_mut(), |b| {
                    b.as_any_mut().downcast_mut::<T>()
                        .expect("AtomData downcast failed — type mismatch (this is a bug in MDDEM)")
                })
            })
    }

    pub fn expect<T: AtomData + 'static>(&self, context: &str) -> Ref<'_, T> {
        self.get::<T>().unwrap_or_else(|| {
            panic!(
                "{}: '{}' not registered. Ensure the plugin is added.",
                context,
                std::any::type_name::<T>()
            )
        })
    }

    pub fn expect_mut<T: AtomData + 'static>(&self, context: &str) -> RefMut<'_, T> {
        self.get_mut::<T>().unwrap_or_else(|| {
            panic!(
                "{}: '{}' not registered. Ensure the plugin is added.",
                context,
                std::any::type_name::<T>()
            )
        })
    }

    /// Truncate all registered extensions to `n` atoms (used when removing ghosts).
    pub fn truncate_all(&self, n: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().truncate(n);
        }
    }

    /// Remove atom at index `i` from all registered extensions via swap-remove.
    pub fn swap_remove_all(&self, i: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().swap_remove(i);
        }
    }

    /// Pack all extension fields for atom `i` into `buf` (exchange/border comm).
    pub fn pack_all(&self, i: usize, buf: &mut Vec<f64>) {
        for (_, cell) in &self.stores {
            cell.borrow().pack(i, buf);
        }
    }

    /// Unpack one atom's extension fields from `buf`. Returns total `f64`s consumed.
    pub fn unpack_all(&self, buf: &[f64]) -> usize {
        let mut pos = 0;
        for (_, cell) in &self.stores {
            pos += cell.borrow_mut().unpack(&buf[pos..]);
        }
        pos
    }

    /// Reorder all extensions according to `perm[0..n]`.
    pub fn apply_permutation_all(&self, perm: &[usize], n: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().apply_permutation(perm, n);
        }
    }

    /// Pack forward-comm fields for atom `i` across all extensions.
    pub fn pack_forward_all(&self, i: usize, buf: &mut Vec<f64>) {
        for (_, cell) in &self.stores {
            cell.borrow().pack_forward(i, buf);
        }
    }

    /// Unpack forward-comm fields for atom `i` across all extensions.
    /// Returns total `f64`s consumed.
    pub fn unpack_forward_all(&self, i: usize, buf: &[f64]) -> usize {
        let mut pos = 0;
        for (_, cell) in &self.stores {
            pos += cell.borrow_mut().unpack_forward(i, &buf[pos..]);
        }
        pos
    }

    /// Pack reverse-comm fields (e.g. torque) for atom `i` across all extensions.
    pub fn pack_reverse_all(&self, i: usize, buf: &mut Vec<f64>) {
        for (_, cell) in &self.stores {
            cell.borrow().pack_reverse(i, buf);
        }
    }

    /// Unpack reverse-comm fields for atom `i` across all extensions.
    /// Returns total `f64`s consumed.
    pub fn unpack_reverse_all(&self, i: usize, buf: &[f64]) -> usize {
        let mut pos = 0;
        for (_, cell) in &self.stores {
            pos += cell.borrow_mut().unpack_reverse(i, &buf[pos..]);
        }
        pos
    }

    /// Zero per-step accumulators (e.g. torque) in all extensions for atoms `0..n`.
    pub fn zero_all(&self, n: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().zero(n);
        }
    }

    /// Total number of `f64`s per atom in forward comm across all extensions.
    pub fn forward_comm_size(&self) -> usize {
        self.stores.iter().map(|(_, cell)| cell.borrow().forward_comm_size()).sum()
    }

    /// Total number of `f64`s per atom in reverse comm across all extensions.
    pub fn reverse_comm_size(&self) -> usize {
        self.stores.iter().map(|(_, cell)| cell.borrow().reverse_comm_size()).sum()
    }

    /// Pack all local atoms for restart file serialization.
    /// Returns one `Vec<f64>` per registered extension.
    pub fn pack_all_for_restart(&self, nlocal: usize) -> Vec<Vec<f64>> {
        self.stores
            .iter()
            .map(|(_, cell)| {
                let store = cell.borrow();
                let mut buf = Vec::new();
                for i in 0..nlocal {
                    store.pack(i, &mut buf);
                }
                buf
            })
            .collect()
    }

    /// Restore all extensions from restart file buffers (one buffer per extension).
    pub fn unpack_all_from_restart(&self, buffers: &[Vec<f64>]) {
        for ((_, cell), buf) in self.stores.iter().zip(buffers.iter()) {
            let mut store = cell.borrow_mut();
            let mut pos = 0;
            while pos < buf.len() {
                pos += store.unpack(&buf[pos..]);
            }
        }
    }
}

// ── Atom Vec field macro ─────────────────────────────────────────────────────

/// Enumerates all per-atom Vec fields with their element types.
/// Pass a callback macro name; it receives the full list as
/// `(field, Type), ...` and can generate code uniformly.
#[macro_export]
macro_rules! for_each_atom_vec {
    ($callback:ident) => {
        $callback! {
            (tag, u32),
            (atom_type, u32),
            (origin_index, i32),
            (is_ghost, bool),
            (pos, [f64; 3]),
            (vel, [f64; 3]),
            (force, [f64; 3]),
            (cutoff_radius, f64),
            (mass, f64),
            (inv_mass, f64),
            (image, [i32; 3]),
        }
    };
}

// ── Atom ──────────────────────────────────────────────────────────────────────

/// Struct-of-arrays storage for all per-atom fields (position, velocity, force, etc.).
pub struct Atom {
    pub natoms: u64,
    pub nlocal: u32,
    pub nghost: u32,

    /// Number of distinct atom types in the simulation.
    pub ntypes: usize,

    pub dt: f64,

    pub tag: Vec<u32>,
    pub atom_type: Vec<u32>,
    pub origin_index: Vec<i32>,
    pub is_ghost: Vec<bool>,

    // Interleaved arrays: field[i] = [x, y, z]
    pub pos: Vec<[f64; 3]>,
    pub vel: Vec<[f64; 3]>,
    pub force: Vec<[f64; 3]>,

    pub cutoff_radius: Vec<f64>,
    pub mass: Vec<f64>,
    pub inv_mass: Vec<f64>,
    /// PBC image flags: number of times atom has crossed each periodic boundary.
    pub image: Vec<[i32; 3]>,
}

impl Default for Atom {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! impl_atom_new {
    ( $( ($field:ident, $ty:ty) ),* $(,)? ) => {
        pub fn new() -> Self {
            Atom {
                natoms: 0,
                nlocal: 0,
                nghost: 0,
                ntypes: 1,
                dt: 1.0,
                $( $field: Vec::new(), )*
            }
        }
    };
}

macro_rules! impl_atom_swap_remove {
    ( $( ($field:ident, $ty:ty) ),* $(,)? ) => {
        pub fn swap_remove(&mut self, i: usize) {
            $( self.$field.swap_remove(i); )*
        }
    };
}

macro_rules! impl_atom_truncate {
    ( $( ($field:ident, $ty:ty) ),* $(,)? ) => {
        pub fn truncate_to_nlocal(&mut self) {
            let n = self.nlocal as usize;
            $( self.$field.truncate(n); )*
        }
    };
}

macro_rules! impl_atom_reserve {
    ( $( ($field:ident, $ty:ty) ),* $(,)? ) => {
        pub fn reserve(&mut self, additional: usize) {
            $( self.$field.reserve(additional); )*
        }
    };
}

macro_rules! impl_atom_apply_permutation {
    ( $( ($field:ident, $ty:ty) ),* $(,)? ) => {
        pub fn apply_permutation(&mut self, perm: &[usize], n: usize) {
            $(
                {
                    let scratch: Vec<$ty> = perm.iter().map(|&p| self.$field[p].clone()).collect();
                    self.$field[..n].clone_from_slice(&scratch);
                }
            )*
        }
    };
}

impl Atom {
    for_each_atom_vec!(impl_atom_new);
    for_each_atom_vec!(impl_atom_swap_remove);
    for_each_atom_vec!(impl_atom_truncate);
    for_each_atom_vec!(impl_atom_reserve);
    for_each_atom_vec!(impl_atom_apply_permutation);

    /// Total number of atoms (local + ghost) currently stored.
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    /// Returns true if no atoms are stored.
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// Dimension-indexed position access for comm.rs border detection.
    pub fn pos_component(&self, i: usize, dim: usize) -> f64 {
        self.pos[i][dim]
    }

    /// Returns the highest global tag among all stored atoms, or 0 if empty.
    pub fn get_max_tag(&self) -> u32 {
        self.tag.iter().cloned().max().unwrap_or(0)
    }

    /// Shared packing logic: writes [`ATOM_PACK_SIZE`] `f64`s for atom `i` into `buf`,
    /// with caller-supplied origin_index and position offset.
    fn pack_atom_inner(&self, i: usize, origin_index_val: f64, pos_offset: [f64; 3], buf: &mut Vec<f64>) {
        buf.push(self.tag[i] as f64);
        buf.push(origin_index_val);
        buf.push(self.cutoff_radius[i]);
        buf.push(self.atom_type[i] as f64);
        buf.push(self.pos[i][0] + pos_offset[0]);
        buf.push(self.pos[i][1] + pos_offset[1]);
        buf.push(self.pos[i][2] + pos_offset[2]);
        buf.push(self.vel[i][0]);
        buf.push(self.vel[i][1]);
        buf.push(self.vel[i][2]);
        buf.push(self.force[i][0]);
        buf.push(self.force[i][1]);
        buf.push(self.force[i][2]);
        buf.push(self.mass[i]);
        buf.push(self.image[i][0] as f64);
        buf.push(self.image[i][1] as f64);
        buf.push(self.image[i][2] as f64);
    }

    /// Pack atom `i` for MPI exchange (migration to another rank). Sets origin_index to 0.
    pub fn pack_exchange(&self, i: usize, buf: &mut Vec<f64>) {
        self.pack_atom_inner(i, 0.0, [0.0; 3], buf);
    }

    /// Pack atom `i` as a ghost for border communication, applying a periodic position shift.
    /// Stores the local index as `origin_index` so reverse comm can accumulate forces back.
    pub fn pack_border(&mut self, i: usize, change_pos: [f64; 3], buf: &mut Vec<f64>) {
        self.pack_atom_inner(i, i as f64, change_pos, buf);
    }

    /// Unpack one atom from `buf` (ghost or exchanged). Returns [`ATOM_PACK_SIZE`].
    pub fn unpack_atom(&mut self, buf: &[f64], is_ghost: bool) -> usize {
        self.tag.push(buf[0] as u32);
        self.origin_index.push(buf[1] as i32);
        self.cutoff_radius.push(buf[2]);
        self.atom_type.push(buf[3] as u32);
        self.pos.push([buf[4], buf[5], buf[6]]);
        self.vel.push([buf[7], buf[8], buf[9]]);
        self.force.push([buf[10], buf[11], buf[12]]);
        self.mass.push(buf[13]);
        self.inv_mass.push(1.0 / buf[13]);
        self.image.push([buf[14] as i32, buf[15] as i32, buf[16] as i32]);
        self.is_ghost.push(is_ghost);
        ATOM_PACK_SIZE
    }

    /// Convenience method for tests: push a local atom with default velocity/force.
    pub fn push_test_atom(&mut self, tag: u32, pos: [f64; 3], radius: f64, mass: f64) {
        self.tag.push(tag);
        self.atom_type.push(0);
        self.origin_index.push(0);
        self.pos.push(pos);
        self.vel.push([0.0; 3]);
        self.force.push([0.0; 3]);
        self.mass.push(mass);
        self.inv_mass.push(1.0 / mass);
        self.cutoff_radius.push(radius);
        self.image.push([0, 0, 0]);
        self.is_ghost.push(false);
    }
}

/// Compute kinetic energy over local atoms, optionally filtered by a group mask.
pub fn compute_ke(atoms: &Atom, mask: Option<&[bool]>) -> f64 {
    let nlocal = atoms.nlocal as usize;
    let mut ke = 0.0;
    for i in 0..nlocal {
        if let Some(m) = mask {
            if !m[i] {
                continue;
            }
        }
        let v = atoms.vel[i];
        ke += atoms.mass[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    0.5 * ke
}

// ── Plugin & systems ──────────────────────────────────────────────────────────

/// Registers the [`Atom`] and [`AtomDataRegistry`] resources and per-step force zeroing.
pub struct AtomPlugin;

impl Plugin for AtomPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Atom::new())
            .add_resource(AtomDataRegistry::new())
            .add_update_system(
                remove_ghost_atoms.run_if(in_state(CommState::FullRebuild)),
                ParticleSimScheduleSet::PostInitialIntegration,
            )
            .add_update_system(zero_all_forces, ParticleSimScheduleSet::PostInitialIntegration);
    }
}

/// Truncate ghost atoms before the next communication phase.
/// Gated by `run_if(in_state(CommState::FullRebuild))` — only runs on full rebuild steps.
pub fn remove_ghost_atoms(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    atoms.truncate_to_nlocal();
    registry.truncate_all(atoms.nlocal as usize);
    atoms.nghost = 0;
}

fn zero_all_forces(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    let n = atoms.len();
    atoms.force[..n].fill([0.0; 3]);
    registry.zero_all(n);
}

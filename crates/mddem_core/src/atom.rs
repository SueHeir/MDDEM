use std::{
    any::{Any, TypeId},
    cell::{Ref, RefCell, RefMut},
};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use nalgebra::Vector3;

/// Number of f64s packed/unpacked for one atom's base fields.
pub const ATOM_PACK_SIZE: usize = 15;

// ── AtomData trait ───────────────────────────────────────────────────────────

/// Per-atom extension data (e.g. radius, density). Supports pack/unpack for MPI communication.
pub trait AtomData: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn truncate(&mut self, n: usize);
    fn swap_remove(&mut self, i: usize);
    fn pack(&self, i: usize, buf: &mut Vec<f64>);
    fn unpack(&mut self, buf: &[f64]) -> usize;
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

    pub fn register<T: AtomData + 'static>(&mut self, data: T) {
        let id = TypeId::of::<T>();
        for (existing_id, _) in &self.stores {
            if *existing_id == id {
                panic!("AtomData type already registered");
            }
        }
        self.stores.push((id, RefCell::new(Box::new(data))));
    }

    pub fn get<T: AtomData + 'static>(&self) -> Option<Ref<'_, T>> {
        let id = TypeId::of::<T>();
        self.stores
            .iter()
            .find(|(tid, _)| *tid == id)
            .map(|(_, cell)| Ref::map(cell.borrow(), |b| b.as_any().downcast_ref::<T>().unwrap()))
    }

    pub fn get_mut<T: AtomData + 'static>(&self) -> Option<RefMut<'_, T>> {
        let id = TypeId::of::<T>();
        self.stores
            .iter()
            .find(|(tid, _)| *tid == id)
            .map(|(_, cell)| {
                RefMut::map(cell.borrow_mut(), |b| {
                    b.as_any_mut().downcast_mut::<T>().unwrap()
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

    pub fn truncate_all(&self, n: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().truncate(n);
        }
    }

    pub fn swap_remove_all(&self, i: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().swap_remove(i);
        }
    }

    pub fn pack_all(&self, i: usize, buf: &mut Vec<f64>) {
        for (_, cell) in &self.stores {
            cell.borrow().pack(i, buf);
        }
    }

    pub fn unpack_all(&self, buf: &[f64]) -> usize {
        let mut pos = 0;
        for (_, cell) in &self.stores {
            pos += cell.borrow_mut().unpack(&buf[pos..]);
        }
        pos
    }

    pub fn apply_permutation_all(&self, perm: &[usize], n: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().apply_permutation(perm, n);
        }
    }

    pub fn pack_forward_all(&self, i: usize, buf: &mut Vec<f64>) {
        for (_, cell) in &self.stores {
            cell.borrow().pack_forward(i, buf);
        }
    }

    pub fn unpack_forward_all(&self, i: usize, buf: &[f64]) -> usize {
        let mut pos = 0;
        for (_, cell) in &self.stores {
            pos += cell.borrow_mut().unpack_forward(i, &buf[pos..]);
        }
        pos
    }

    pub fn pack_reverse_all(&self, i: usize, buf: &mut Vec<f64>) {
        for (_, cell) in &self.stores {
            cell.borrow().pack_reverse(i, buf);
        }
    }

    pub fn unpack_reverse_all(&self, i: usize, buf: &[f64]) -> usize {
        let mut pos = 0;
        for (_, cell) in &self.stores {
            pos += cell.borrow_mut().unpack_reverse(i, &buf[pos..]);
        }
        pos
    }

    pub fn zero_all(&self, n: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().zero(n);
        }
    }

    pub fn forward_comm_size(&self) -> usize {
        self.stores.iter().map(|(_, cell)| cell.borrow().forward_comm_size()).sum()
    }

    pub fn reverse_comm_size(&self) -> usize {
        self.stores.iter().map(|(_, cell)| cell.borrow().reverse_comm_size()).sum()
    }

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
            (is_collision, bool),
            (pos, [f64; 3]),
            (vel, [f64; 3]),
            (force, [f64; 3]),
            (skin, f64),
            (mass, f64),
            (inv_mass, f64),
        }
    };
}

// ── Atom ──────────────────────────────────────────────────────────────────────

/// Struct-of-arrays storage for all per-atom fields (position, velocity, force, etc.).
pub struct Atom {
    pub natoms: u64,
    pub nlocal: u32,
    pub nghost: u32,

    pub dt: f64,

    /// When true, ghost ordering is stable (single-process deterministic iteration),
    /// so the neighbor list remains valid across ghost rebuilds. In MPI mode this
    /// stays false, forcing neighbor rebuild every step.
    pub communicate_only: bool,

    /// When true, PBC boundary crossings force a full ghost + neighbor rebuild.
    /// Required for DEM where stale ghost placement causes missed contacts.
    pub rebuild_on_pbc_wrap: bool,

    pub tag: Vec<u32>,
    pub atom_type: Vec<u32>,
    pub origin_index: Vec<i32>,
    pub is_ghost: Vec<bool>,
    pub is_collision: Vec<bool>,

    // Interleaved arrays: field[i] = [x, y, z]
    pub pos: Vec<[f64; 3]>,
    pub vel: Vec<[f64; 3]>,
    pub force: Vec<[f64; 3]>,

    pub skin: Vec<f64>,
    pub mass: Vec<f64>,
    pub inv_mass: Vec<f64>,
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
                dt: 1.0,
                communicate_only: false,
                rebuild_on_pbc_wrap: false,
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

    pub fn get_max_tag(&self) -> u32 {
        self.tag.iter().cloned().max().unwrap_or(0)
    }

    fn pack_atom_inner(&self, i: usize, origin_index_val: f64, pos_offset: Vector3<f64>, buf: &mut Vec<f64>) {
        buf.push(self.tag[i] as f64);
        buf.push(origin_index_val);
        buf.push(self.skin[i]);
        buf.push(self.atom_type[i] as f64);
        buf.push(self.pos[i][0] + pos_offset.x);
        buf.push(self.pos[i][1] + pos_offset.y);
        buf.push(self.pos[i][2] + pos_offset.z);
        buf.push(self.vel[i][0]);
        buf.push(self.vel[i][1]);
        buf.push(self.vel[i][2]);
        buf.push(self.force[i][0]);
        buf.push(self.force[i][1]);
        buf.push(self.force[i][2]);
        buf.push(self.mass[i]);
        buf.push(if self.is_collision[i] { 0.0 } else { 1.0 });
    }

    pub fn pack_exchange(&self, i: usize, buf: &mut Vec<f64>) {
        self.pack_atom_inner(i, 0.0, Vector3::zeros(), buf);
    }

    pub fn pack_border(&mut self, i: usize, change_pos: Vector3<f64>, buf: &mut Vec<f64>) {
        self.pack_atom_inner(i, i as f64, change_pos, buf);
    }

    pub fn unpack_atom(&mut self, buf: &[f64], is_ghost: bool) -> usize {
        self.tag.push(buf[0] as u32);
        self.origin_index.push(buf[1] as i32);
        self.skin.push(buf[2]);
        self.atom_type.push(buf[3] as u32);
        self.pos.push([buf[4], buf[5], buf[6]]);
        self.vel.push([buf[7], buf[8], buf[9]]);
        self.force.push([buf[10], buf[11], buf[12]]);
        self.mass.push(buf[13]);
        self.inv_mass.push(1.0 / buf[13]);
        self.is_collision.push(buf[14] == 0.0);
        self.is_ghost.push(is_ghost);
        ATOM_PACK_SIZE
    }

    pub fn push_test_atom(&mut self, tag: u32, pos: Vector3<f64>, radius: f64, mass: f64) {
        self.tag.push(tag);
        self.atom_type.push(0);
        self.origin_index.push(0);
        self.pos.push([pos.x, pos.y, pos.z]);
        self.vel.push([0.0; 3]);
        self.force.push([0.0; 3]);
        self.mass.push(mass);
        self.inv_mass.push(1.0 / mass);
        self.skin.push(radius);
        self.is_ghost.push(false);
        self.is_collision.push(false);
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
            .add_update_system(remove_ghost_atoms.label("remove_ghost_atoms"), ScheduleSet::PostInitialIntegration)
            .add_update_system(zero_all_forces, ScheduleSet::PostInitialIntegration);
    }
}

fn remove_ghost_atoms(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    if atoms.communicate_only {
        return;
    }
    atoms.truncate_to_nlocal();
    registry.truncate_all(atoms.nlocal as usize);
    atoms.nghost = 0;
}

fn zero_all_forces(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    let n = atoms.len();
    atoms.is_collision[..n].fill(false);
    atoms.force[..n].fill([0.0; 3]);
    registry.zero_all(n);
}

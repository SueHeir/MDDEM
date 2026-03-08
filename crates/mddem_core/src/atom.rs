use std::{
    any::{Any, TypeId},
    cell::{Ref, RefCell, RefMut},
    ops::{Index, IndexMut},
};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
#[cfg(feature = "mpi_backend")]
use mpi::traits::Equivalence;
use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// Number of f64s packed/unpacked for one atom's base fields.
pub const ATOM_PACK_SIZE: usize = 28;

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

// ── MPI helper types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "mpi_backend", derive(Equivalence))]
pub struct Vector3f64MPI {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Index<&'_ usize> for Vector3f64MPI {
    type Output = f64;
    fn index(&self, s: &usize) -> &f64 {
        match s {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("unknown field: {}", s),
        }
    }
}

impl IndexMut<&'_ usize> for Vector3f64MPI {
    fn index_mut(&mut self, s: &usize) -> &mut f64 {
        match s {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("unknown field: {}", s),
        }
    }
}

impl Vector3f64MPI {
    pub fn new_from_components(x: f64, y: f64, z: f64) -> Vector3f64MPI {
        Vector3f64MPI { x, y, z }
    }

    pub fn to_vector3(self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "mpi_backend", derive(Equivalence))]
pub struct Quaternionf64MPI {
    pub i: f64,
    pub j: f64,
    pub k: f64,
    pub w: f64,
}

impl Quaternionf64MPI {
    pub fn new(vec: UnitQuaternion<f64>) -> Quaternionf64MPI {
        Quaternionf64MPI {
            i: vec.i,
            j: vec.j,
            k: vec.k,
            w: vec.w,
        }
    }

    pub fn to_quat(self) -> UnitQuaternion<f64> {
        UnitQuaternion::from_quaternion(Quaternion::new(self.w, self.i, self.j, self.k))
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "mpi_backend", derive(Equivalence))]
/// MPI-serializable force + torque payload for ghost → local atom reduction.
pub struct ForceMPI {
    pub tag: u32,
    pub origin_index: u32,
    pub torque: Vector3f64MPI,
    pub force: Vector3f64MPI,
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
            (pos_x, f64),
            (pos_y, f64),
            (pos_z, f64),
            (vel_x, f64),
            (vel_y, f64),
            (vel_z, f64),
            (quaternion, nalgebra::UnitQuaternion<f64>),
            (omega_x, f64),
            (omega_y, f64),
            (omega_z, f64),
            (ang_mom_x, f64),
            (ang_mom_y, f64),
            (ang_mom_z, f64),
            (torque_x, f64),
            (torque_y, f64),
            (torque_z, f64),
            (force_x, f64),
            (force_y, f64),
            (force_z, f64),
            (skin, f64),
            (mass, f64),
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

    pub tag: Vec<u32>,
    pub atom_type: Vec<u32>,
    pub origin_index: Vec<i32>,
    pub is_ghost: Vec<bool>,
    pub is_collision: Vec<bool>,

    // SoA position
    pub pos_x: Vec<f64>,
    pub pos_y: Vec<f64>,
    pub pos_z: Vec<f64>,
    // SoA velocity
    pub vel_x: Vec<f64>,
    pub vel_y: Vec<f64>,
    pub vel_z: Vec<f64>,
    // Quaternion (kept as-is, not in hot loops)
    pub quaternion: Vec<UnitQuaternion<f64>>,
    // SoA angular velocity
    pub omega_x: Vec<f64>,
    pub omega_y: Vec<f64>,
    pub omega_z: Vec<f64>,
    // SoA angular momentum
    pub ang_mom_x: Vec<f64>,
    pub ang_mom_y: Vec<f64>,
    pub ang_mom_z: Vec<f64>,
    // SoA torque
    pub torque_x: Vec<f64>,
    pub torque_y: Vec<f64>,
    pub torque_z: Vec<f64>,
    // SoA force
    pub force_x: Vec<f64>,
    pub force_y: Vec<f64>,
    pub force_z: Vec<f64>,

    pub skin: Vec<f64>,
    pub mass: Vec<f64>,
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
        self.pos_x.len()
    }

    /// Returns true if no atoms are stored.
    pub fn is_empty(&self) -> bool {
        self.pos_x.is_empty()
    }

    /// Dimension-indexed position access for comm.rs border detection.
    pub fn pos_component(&self, i: usize, dim: usize) -> f64 {
        match dim {
            0 => self.pos_x[i],
            1 => self.pos_y[i],
            2 => self.pos_z[i],
            _ => panic!("invalid dimension {}", dim),
        }
    }

    pub fn get_max_tag(&self) -> u32 {
        self.tag.iter().cloned().max().unwrap_or(0)
    }

    pub fn get_force_data(&mut self, i: usize) -> ForceMPI {
        debug_assert!(self.is_ghost[i], "get_force_data called on non-ghost atom {}", i);
        ForceMPI {
            tag: self.tag[i],
            origin_index: self.origin_index[i] as u32,
            torque: Vector3f64MPI::new_from_components(
                self.torque_x[i],
                self.torque_y[i],
                self.torque_z[i],
            ),
            force: Vector3f64MPI::new_from_components(
                self.force_x[i],
                self.force_y[i],
                self.force_z[i],
            ),
        }
    }

    pub fn apply_force_data(&mut self, force_mpi: ForceMPI) {
        let i = force_mpi.origin_index as usize;
        debug_assert_eq!(
            force_mpi.tag, self.tag[i],
            "apply_force_data: force tag {} != atom tag {} at index {}",
            force_mpi.tag, self.tag[i], i
        );
        let f = force_mpi.force;
        self.force_x[i] += f.x;
        self.force_y[i] += f.y;
        self.force_z[i] += f.z;
        let t = force_mpi.torque;
        self.torque_x[i] += t.x;
        self.torque_y[i] += t.y;
        self.torque_z[i] += t.z;
    }

    pub fn pack_exchange(&self, i: usize, buf: &mut Vec<f64>) {
        buf.push(self.tag[i] as f64);
        buf.push(0.0);
        buf.push(self.skin[i]);
        buf.push(self.atom_type[i] as f64);
        buf.push(self.pos_x[i]);
        buf.push(self.pos_y[i]);
        buf.push(self.pos_z[i]);
        buf.push(self.vel_x[i]);
        buf.push(self.vel_y[i]);
        buf.push(self.vel_z[i]);
        let q = self.quaternion[i];
        buf.push(q.w);
        buf.push(q.i);
        buf.push(q.j);
        buf.push(q.k);
        buf.push(self.omega_x[i]);
        buf.push(self.omega_y[i]);
        buf.push(self.omega_z[i]);
        buf.push(self.ang_mom_x[i]);
        buf.push(self.ang_mom_y[i]);
        buf.push(self.ang_mom_z[i]);
        buf.push(self.torque_x[i]);
        buf.push(self.torque_y[i]);
        buf.push(self.torque_z[i]);
        buf.push(self.force_x[i]);
        buf.push(self.force_y[i]);
        buf.push(self.force_z[i]);
        buf.push(self.mass[i]);
        buf.push(if self.is_collision[i] { 0.0 } else { 1.0 });
    }

    pub fn pack_border(&mut self, i: usize, change_pos: Vector3<f64>, buf: &mut Vec<f64>) {
        buf.push(self.tag[i] as f64);
        buf.push(i as f64);
        buf.push(self.skin[i]);
        buf.push(self.atom_type[i] as f64);
        buf.push(self.pos_x[i] + change_pos.x);
        buf.push(self.pos_y[i] + change_pos.y);
        buf.push(self.pos_z[i] + change_pos.z);
        buf.push(self.vel_x[i]);
        buf.push(self.vel_y[i]);
        buf.push(self.vel_z[i]);
        let q = self.quaternion[i];
        buf.push(q.w);
        buf.push(q.i);
        buf.push(q.j);
        buf.push(q.k);
        buf.push(self.omega_x[i]);
        buf.push(self.omega_y[i]);
        buf.push(self.omega_z[i]);
        buf.push(self.ang_mom_x[i]);
        buf.push(self.ang_mom_y[i]);
        buf.push(self.ang_mom_z[i]);
        buf.push(self.torque_x[i]);
        buf.push(self.torque_y[i]);
        buf.push(self.torque_z[i]);
        buf.push(self.force_x[i]);
        buf.push(self.force_y[i]);
        buf.push(self.force_z[i]);
        buf.push(self.mass[i]);
        buf.push(if self.is_collision[i] { 0.0 } else { 1.0 });
    }

    pub fn unpack_atom(&mut self, buf: &[f64], is_ghost: bool) -> usize {
        self.tag.push(buf[0] as u32);
        self.origin_index.push(buf[1] as i32);
        self.skin.push(buf[2]);
        self.atom_type.push(buf[3] as u32);
        self.pos_x.push(buf[4]);
        self.pos_y.push(buf[5]);
        self.pos_z.push(buf[6]);
        self.vel_x.push(buf[7]);
        self.vel_y.push(buf[8]);
        self.vel_z.push(buf[9]);
        self.quaternion
            .push(UnitQuaternion::from_quaternion(Quaternion::new(
                buf[10], buf[11], buf[12], buf[13],
            )));
        self.omega_x.push(buf[14]);
        self.omega_y.push(buf[15]);
        self.omega_z.push(buf[16]);
        self.ang_mom_x.push(buf[17]);
        self.ang_mom_y.push(buf[18]);
        self.ang_mom_z.push(buf[19]);
        self.torque_x.push(buf[20]);
        self.torque_y.push(buf[21]);
        self.torque_z.push(buf[22]);
        self.force_x.push(buf[23]);
        self.force_y.push(buf[24]);
        self.force_z.push(buf[25]);
        self.mass.push(buf[26]);
        self.is_collision.push(buf[27] == 0.0);
        self.is_ghost.push(is_ghost);
        ATOM_PACK_SIZE
    }

    pub fn push_test_atom(&mut self, tag: u32, pos: Vector3<f64>, radius: f64, mass: f64) {
        self.tag.push(tag);
        self.atom_type.push(0);
        self.origin_index.push(0);
        self.pos_x.push(pos.x);
        self.pos_y.push(pos.y);
        self.pos_z.push(pos.z);
        self.vel_x.push(0.0);
        self.vel_y.push(0.0);
        self.vel_z.push(0.0);
        self.force_x.push(0.0);
        self.force_y.push(0.0);
        self.force_z.push(0.0);
        self.torque_x.push(0.0);
        self.torque_y.push(0.0);
        self.torque_z.push(0.0);
        self.mass.push(mass);
        self.skin.push(radius);
        self.is_ghost.push(false);
        self.is_collision.push(false);
        self.quaternion.push(UnitQuaternion::identity());
        self.omega_x.push(0.0);
        self.omega_y.push(0.0);
        self.omega_z.push(0.0);
        self.ang_mom_x.push(0.0);
        self.ang_mom_y.push(0.0);
        self.ang_mom_z.push(0.0);
    }
}

// ── Plugin & systems ──────────────────────────────────────────────────────────

/// Registers the [`Atom`] and [`AtomDataRegistry`] resources and per-step force zeroing.
pub struct AtomPlugin;

impl Plugin for AtomPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Atom::new())
            .add_resource(AtomDataRegistry::new())
            .add_update_system(remove_ghost_atoms, ScheduleSet::PostInitialIntegration)
            .add_update_system(zero_all_forces, ScheduleSet::PostInitialIntegration);
    }
}

fn remove_ghost_atoms(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    atoms.truncate_to_nlocal();
    registry.truncate_all(atoms.nlocal as usize);
    atoms.nghost = 0;
}

fn zero_all_forces(mut atoms: ResMut<Atom>) {
    for i in 0..atoms.len() {
        atoms.is_collision[i] = false;
        atoms.force_x[i] = 0.0;
        atoms.force_y[i] = 0.0;
        atoms.force_z[i] = 0.0;
        atoms.torque_x[i] = 0.0;
        atoms.torque_y[i] = 0.0;
        atoms.torque_z[i] = 0.0;
    }
}

use std::{
    any::{Any, TypeId},
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
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

pub trait AtomData: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn truncate(&mut self, n: usize);
    fn swap_remove(&mut self, i: usize);
    fn pack(&self, i: usize, buf: &mut Vec<f64>);
    fn unpack(&mut self, buf: &[f64]) -> usize;
}


// ── AtomDataRegistry ─────────────────────────────────────────────────────────

pub struct AtomDataRegistry {
    stores: HashMap<TypeId, RefCell<Box<dyn AtomData>>>,
}

impl AtomDataRegistry {
    pub fn new() -> Self {
        AtomDataRegistry { stores: HashMap::new() }
    }

    pub fn register<T: AtomData + 'static>(&mut self, data: T) {
        self.stores.insert(TypeId::of::<T>(), RefCell::new(Box::new(data)));
    }

    pub fn get<T: AtomData + 'static>(&self) -> Option<Ref<'_, T>> {
        self.stores.get(&TypeId::of::<T>()).map(|cell| {
            Ref::map(cell.borrow(), |b| {
                b.as_any().downcast_ref::<T>().unwrap()
            })
        })
    }

    pub fn get_mut<T: AtomData + 'static>(&self) -> Option<RefMut<'_, T>> {
        self.stores.get(&TypeId::of::<T>()).map(|cell| {
            RefMut::map(cell.borrow_mut(), |b| {
                b.as_any_mut().downcast_mut::<T>().unwrap()
            })
        })
    }

    pub fn truncate_all(&self, n: usize) {
        for cell in self.stores.values() {
            cell.borrow_mut().truncate(n);
        }
    }

    pub fn swap_remove_all(&self, i: usize) {
        for cell in self.stores.values() {
            cell.borrow_mut().swap_remove(i);
        }
    }

    pub fn pack_all(&self, i: usize, buf: &mut Vec<f64>) {
        for cell in self.stores.values() {
            cell.borrow().pack(i, buf);
        }
    }

    pub fn unpack_all(&self, buf: &[f64]) -> usize {
        let mut pos = 0;
        for cell in self.stores.values() {
            pos += cell.borrow_mut().unpack(&buf[pos..]);
        }
        pos
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
        Quaternionf64MPI { i: vec.i, j: vec.j, k: vec.k, w: vec.w }
    }

    pub fn to_quat(self) -> UnitQuaternion<f64> {
        UnitQuaternion::from_quaternion(Quaternion::new(self.w, self.i, self.j, self.k))
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "mpi_backend", derive(Equivalence))]
pub struct ForceMPI {
    pub tag: u32,
    pub origin_index: u32,
    pub torque: Vector3f64MPI,
    pub force: Vector3f64MPI,
}


// ── Atom ──────────────────────────────────────────────────────────────────────

pub struct Atom {
    pub natoms: u64,
    pub nlocal: u32,
    pub nghost: u32,

    pub dt: f64,

    pub tag: Vec<u32>,
    pub atom_type: Vec<u32>,
    pub origin_index: Vec<i32>,
    pub is_ghost: Vec<bool>,
    pub has_ghost: Vec<bool>,
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
    pub quaterion: Vec<UnitQuaternion<f64>>,
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

impl Atom {
    pub fn new() -> Self {
        Atom {
            natoms: 0,
            nlocal: 0,
            nghost: 0,
            dt: 1.0,
            tag: Vec::new(),
            atom_type: Vec::new(),
            origin_index: Vec::new(),
            is_ghost: Vec::new(),
            has_ghost: Vec::new(),
            is_collision: Vec::new(),
            pos_x: Vec::new(), pos_y: Vec::new(), pos_z: Vec::new(),
            vel_x: Vec::new(), vel_y: Vec::new(), vel_z: Vec::new(),
            quaterion: Vec::new(),
            omega_x: Vec::new(), omega_y: Vec::new(), omega_z: Vec::new(),
            ang_mom_x: Vec::new(), ang_mom_y: Vec::new(), ang_mom_z: Vec::new(),
            torque_x: Vec::new(), torque_y: Vec::new(), torque_z: Vec::new(),
            force_x: Vec::new(), force_y: Vec::new(), force_z: Vec::new(),
            skin: Vec::new(),
            mass: Vec::new(),
        }
    }

    /// Total number of atoms (local + ghost) currently stored.
    pub fn len(&self) -> usize {
        self.pos_x.len()
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
        if !self.is_ghost[i] { panic!(); }
        ForceMPI {
            tag: self.tag[i],
            origin_index: self.origin_index[i] as u32,
            torque: Vector3f64MPI::new_from_components(self.torque_x[i], self.torque_y[i], self.torque_z[i]),
            force: Vector3f64MPI::new_from_components(self.force_x[i], self.force_y[i], self.force_z[i]),
        }
    }

    pub fn apply_force_data(&mut self, force_mpi: ForceMPI) {
        let i = force_mpi.origin_index as usize;
        if force_mpi.tag != self.tag[i] { panic!(); }
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
        buf.push(self.pos_x[i]); buf.push(self.pos_y[i]); buf.push(self.pos_z[i]);
        buf.push(self.vel_x[i]); buf.push(self.vel_y[i]); buf.push(self.vel_z[i]);
        let q = self.quaterion[i];
        buf.push(q.w); buf.push(q.i); buf.push(q.j); buf.push(q.k);
        buf.push(self.omega_x[i]); buf.push(self.omega_y[i]); buf.push(self.omega_z[i]);
        buf.push(self.ang_mom_x[i]); buf.push(self.ang_mom_y[i]); buf.push(self.ang_mom_z[i]);
        buf.push(self.torque_x[i]); buf.push(self.torque_y[i]); buf.push(self.torque_z[i]);
        buf.push(self.force_x[i]); buf.push(self.force_y[i]); buf.push(self.force_z[i]);
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
        buf.push(self.vel_x[i]); buf.push(self.vel_y[i]); buf.push(self.vel_z[i]);
        let q = self.quaterion[i];
        buf.push(q.w); buf.push(q.i); buf.push(q.j); buf.push(q.k);
        buf.push(self.omega_x[i]); buf.push(self.omega_y[i]); buf.push(self.omega_z[i]);
        buf.push(self.ang_mom_x[i]); buf.push(self.ang_mom_y[i]); buf.push(self.ang_mom_z[i]);
        buf.push(self.torque_x[i]); buf.push(self.torque_y[i]); buf.push(self.torque_z[i]);
        buf.push(self.force_x[i]); buf.push(self.force_y[i]); buf.push(self.force_z[i]);
        buf.push(self.mass[i]);
        buf.push(if self.is_collision[i] { 0.0 } else { 1.0 });
        self.has_ghost[i] = true;
    }

    pub fn unpack_atom(&mut self, buf: &[f64], is_ghost: bool) -> usize {
        self.tag.push(buf[0] as u32);
        self.origin_index.push(buf[1] as i32);
        self.skin.push(buf[2]);
        self.atom_type.push(buf[3] as u32);
        self.pos_x.push(buf[4]); self.pos_y.push(buf[5]); self.pos_z.push(buf[6]);
        self.vel_x.push(buf[7]); self.vel_y.push(buf[8]); self.vel_z.push(buf[9]);
        self.quaterion.push(UnitQuaternion::from_quaternion(
            Quaternion::new(buf[10], buf[11], buf[12], buf[13])
        ));
        self.omega_x.push(buf[14]); self.omega_y.push(buf[15]); self.omega_z.push(buf[16]);
        self.ang_mom_x.push(buf[17]); self.ang_mom_y.push(buf[18]); self.ang_mom_z.push(buf[19]);
        self.torque_x.push(buf[20]); self.torque_y.push(buf[21]); self.torque_z.push(buf[22]);
        self.force_x.push(buf[23]); self.force_y.push(buf[24]); self.force_z.push(buf[25]);
        self.mass.push(buf[26]);
        self.is_collision.push(buf[27] == 0.0);
        self.is_ghost.push(is_ghost);
        self.has_ghost.push(false);
        ATOM_PACK_SIZE
    }

    pub fn swap_remove(&mut self, i: usize) {
        self.tag.swap_remove(i);
        self.atom_type.swap_remove(i);
        self.origin_index.swap_remove(i);
        self.pos_x.swap_remove(i); self.pos_y.swap_remove(i); self.pos_z.swap_remove(i);
        self.vel_x.swap_remove(i); self.vel_y.swap_remove(i); self.vel_z.swap_remove(i);
        self.quaterion.swap_remove(i);
        self.omega_x.swap_remove(i); self.omega_y.swap_remove(i); self.omega_z.swap_remove(i);
        self.ang_mom_x.swap_remove(i); self.ang_mom_y.swap_remove(i); self.ang_mom_z.swap_remove(i);
        self.torque_x.swap_remove(i); self.torque_y.swap_remove(i); self.torque_z.swap_remove(i);
        self.force_x.swap_remove(i); self.force_y.swap_remove(i); self.force_z.swap_remove(i);
        self.skin.swap_remove(i);
        self.mass.swap_remove(i);
        self.is_collision.swap_remove(i);
        self.is_ghost.swap_remove(i);
        self.has_ghost.swap_remove(i);
    }

    #[cfg(test)]
    pub fn push_test_atom(&mut self, tag: u32, pos: Vector3<f64>, radius: f64, mass: f64) {
        self.tag.push(tag);
        self.atom_type.push(0);
        self.origin_index.push(0);
        self.pos_x.push(pos.x); self.pos_y.push(pos.y); self.pos_z.push(pos.z);
        self.vel_x.push(0.0); self.vel_y.push(0.0); self.vel_z.push(0.0);
        self.force_x.push(0.0); self.force_y.push(0.0); self.force_z.push(0.0);
        self.torque_x.push(0.0); self.torque_y.push(0.0); self.torque_z.push(0.0);
        self.mass.push(mass);
        self.skin.push(radius);
        self.is_ghost.push(false);
        self.has_ghost.push(false);
        self.is_collision.push(false);
        self.quaterion.push(UnitQuaternion::identity());
        self.omega_x.push(0.0); self.omega_y.push(0.0); self.omega_z.push(0.0);
        self.ang_mom_x.push(0.0); self.ang_mom_y.push(0.0); self.ang_mom_z.push(0.0);
    }

    pub fn truncate_to_nlocal(&mut self) {
        let n = self.nlocal as usize;
        self.tag.truncate(n);
        self.atom_type.truncate(n);
        self.origin_index.truncate(n);
        self.pos_x.truncate(n); self.pos_y.truncate(n); self.pos_z.truncate(n);
        self.vel_x.truncate(n); self.vel_y.truncate(n); self.vel_z.truncate(n);
        self.quaterion.truncate(n);
        self.omega_x.truncate(n); self.omega_y.truncate(n); self.omega_z.truncate(n);
        self.ang_mom_x.truncate(n); self.ang_mom_y.truncate(n); self.ang_mom_z.truncate(n);
        self.torque_x.truncate(n); self.torque_y.truncate(n); self.torque_z.truncate(n);
        self.force_x.truncate(n); self.force_y.truncate(n); self.force_z.truncate(n);
        self.skin.truncate(n);
        self.mass.truncate(n);
        self.is_collision.truncate(n);
        self.is_ghost.truncate(n);
        self.has_ghost.truncate(n);
    }
}


// ── Plugin & systems ──────────────────────────────────────────────────────────

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
        atoms.has_ghost[i] = false;
        atoms.force_x[i] = 0.0;
        atoms.force_y[i] = 0.0;
        atoms.force_z[i] = 0.0;
        atoms.torque_x[i] = 0.0;
        atoms.torque_y[i] = 0.0;
        atoms.torque_z[i] = 0.0;
    }
}

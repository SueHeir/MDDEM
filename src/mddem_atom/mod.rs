use std::{
    any::{Any, TypeId},
    cell::{Ref, RefCell, RefMut},
    f64::consts::PI,
    ops::{Index, IndexMut},
};

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use crate::{mddem_communication::Comm, mddem_domain::Domain, mddem_input::Input};

use mpi::traits::{CommunicatorCollectives, Equivalence};
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use rand::Rng;
use rand_distr::{Normal, Distribution};


/// Number of f64s packed/unpacked for one atom's base fields.
pub const ATOM_PACK_SIZE: usize = 27;


// ── AtomData trait ───────────────────────────────────────────────────────────
//
// Implement this for any per-atom data (e.g. DemAtom, MdAtom) that needs to
// travel with atoms through MPI exchange and ghost communication.

pub trait AtomData: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn truncate(&mut self, n: usize);
    fn swap_remove(&mut self, i: usize);
    /// Append atom i's fields to `buf` (non-destructive).
    fn pack(&self, i: usize, buf: &mut Vec<f64>);
    /// Parse one atom's fields from `buf[0..]`, push into vecs.
    /// Returns the number of f64s consumed.
    fn unpack(&mut self, buf: &[f64]) -> usize;
}


// ── AtomDataRegistry ─────────────────────────────────────────────────────────
//
// Separate ECS resource that holds all per-atom extension data.
// Communication systems iterate `stores`; simulation systems access specific
// types via `get` / `get_mut` with zero HashMap overhead.

pub struct AtomDataRegistry {
    stores: Vec<(TypeId, RefCell<Box<dyn AtomData>>)>,
}

impl AtomDataRegistry {
    pub fn new() -> Self {
        AtomDataRegistry { stores: Vec::new() }
    }

    pub fn register<T: AtomData + 'static>(&mut self, data: T) {
        self.stores.push((TypeId::of::<T>(), RefCell::new(Box::new(data))));
    }

    /// Typed immutable borrow. Returns `None` if not registered.
    pub fn get<T: AtomData + 'static>(&self) -> Option<Ref<T>> {
        for (id, cell) in &self.stores {
            if *id == TypeId::of::<T>() {
                return Some(Ref::map(cell.borrow(), |b| {
                    b.as_any().downcast_ref::<T>().unwrap()
                }));
            }
        }
        None
    }

    /// Typed mutable borrow. Returns `None` if not registered.
    pub fn get_mut<T: AtomData + 'static>(&self) -> Option<RefMut<T>> {
        for (id, cell) in &self.stores {
            if *id == TypeId::of::<T>() {
                return Some(RefMut::map(cell.borrow_mut(), |b| {
                    b.as_any_mut().downcast_mut::<T>().unwrap()
                }));
            }
        }
        None
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

    /// Append all extension fields for atom i to `buf`.
    pub fn pack_all(&self, i: usize, buf: &mut Vec<f64>) {
        for (_, cell) in &self.stores {
            cell.borrow().pack(i, buf);
        }
    }

    /// Unpack one atom from each store sequentially.
    /// Returns total number of f64s consumed across all stores.
    pub fn unpack_all(&self, buf: &[f64]) -> usize {
        let mut pos = 0;
        for (_, cell) in &self.stores {
            pos += cell.borrow_mut().unpack(&buf[pos..]);
        }
        pos
    }
}


// ── MPI helper types ──────────────────────────────────────────────────────────

#[derive(Equivalence, Debug, Clone, Copy)]
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
    fn new(vec: Vector3<f64>) -> Vector3f64MPI {
        Vector3f64MPI { x: vec.x, y: vec.y, z: vec.z }
    }

    fn to_vector3(self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }
}

#[derive(Equivalence, Debug, Clone, Copy)]
pub struct Quaternionf64MPI {
    i: f64,
    j: f64,
    k: f64,
    w: f64,
}

impl Quaternionf64MPI {
    fn new(vec: UnitQuaternion<f64>) -> Quaternionf64MPI {
        Quaternionf64MPI { i: vec.i, j: vec.j, k: vec.k, w: vec.w }
    }

    fn to_quat(self) -> UnitQuaternion<f64> {
        UnitQuaternion::from_quaternion(Quaternion::new(self.w, self.i, self.j, self.k))
    }
}

#[derive(Equivalence, Debug, Clone, Copy)]
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
    pub origin_index: Vec<i32>,
    pub is_ghost: Vec<bool>,
    pub has_ghost: Vec<bool>,
    pub is_collision: Vec<bool>,

    pub pos: Vec<Vector3<f64>>,
    pub velocity: Vec<Vector3<f64>>,
    pub quaterion: Vec<UnitQuaternion<f64>>,
    pub omega: Vec<Vector3<f64>>,
    pub angular_momentum: Vec<Vector3<f64>>,
    pub torque: Vec<Vector3<f64>>,
    pub force: Vec<Vector3<f64>>,
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
            origin_index: Vec::new(),
            is_ghost: Vec::new(),
            has_ghost: Vec::new(),
            is_collision: Vec::new(),
            pos: Vec::new(),
            velocity: Vec::new(),
            quaterion: Vec::new(),
            omega: Vec::new(),
            angular_momentum: Vec::new(),
            torque: Vec::new(),
            force: Vec::new(),
            skin: Vec::new(),
            mass: Vec::new(),
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
            torque: Vector3f64MPI::new(self.torque[i]),
            force: Vector3f64MPI::new(self.force[i]),
        }
    }

    pub fn apply_force_data(&mut self, force_mpi: ForceMPI, _rank: i32, _swap: i32, _dim: i32) {
        let i = force_mpi.origin_index as usize;
        if force_mpi.tag != self.tag[i] { panic!(); }
        self.force[i] += force_mpi.force.to_vector3();
        self.torque[i] += force_mpi.torque.to_vector3();
    }

    /// Pack atom i's base fields into `buf` for exchange (atom leaving domain).
    pub fn pack_exchange(&self, i: usize, buf: &mut Vec<f64>) {
        buf.push(self.tag[i] as f64);
        buf.push(0.0); // origin_index placeholder
        buf.push(self.skin[i]);
        let p = self.pos[i];
        buf.push(p.x); buf.push(p.y); buf.push(p.z);
        let v = self.velocity[i];
        buf.push(v.x); buf.push(v.y); buf.push(v.z);
        let q = self.quaterion[i];
        buf.push(q.w); buf.push(q.i); buf.push(q.j); buf.push(q.k);
        let o = self.omega[i];
        buf.push(o.x); buf.push(o.y); buf.push(o.z);
        let am = self.angular_momentum[i];
        buf.push(am.x); buf.push(am.y); buf.push(am.z);
        let t = self.torque[i];
        buf.push(t.x); buf.push(t.y); buf.push(t.z);
        let f = self.force[i];
        buf.push(f.x); buf.push(f.y); buf.push(f.z);
        buf.push(self.mass[i]);
        buf.push(if self.is_collision[i] { 0.0 } else { 1.0 });
    }

    /// Pack atom i as a ghost (border copy), shifting position by `change_pos`.
    /// Also sets `has_ghost[i] = true`.
    pub fn pack_border(&mut self, i: usize, change_pos: Vector3<f64>, buf: &mut Vec<f64>) {
        buf.push(self.tag[i] as f64);
        buf.push(i as f64); // origin_index for reverse-force send
        buf.push(self.skin[i]);
        let p = self.pos[i] + change_pos;
        buf.push(p.x); buf.push(p.y); buf.push(p.z);
        let v = self.velocity[i];
        buf.push(v.x); buf.push(v.y); buf.push(v.z);
        let q = self.quaterion[i];
        buf.push(q.w); buf.push(q.i); buf.push(q.j); buf.push(q.k);
        let o = self.omega[i];
        buf.push(o.x); buf.push(o.y); buf.push(o.z);
        let am = self.angular_momentum[i];
        buf.push(am.x); buf.push(am.y); buf.push(am.z);
        let t = self.torque[i];
        buf.push(t.x); buf.push(t.y); buf.push(t.z);
        let f = self.force[i];
        buf.push(f.x); buf.push(f.y); buf.push(f.z);
        buf.push(self.mass[i]);
        buf.push(if self.is_collision[i] { 0.0 } else { 1.0 });
        self.has_ghost[i] = true;
    }

    /// Unpack one atom from `buf[0..]` and push to vecs.
    /// Returns ATOM_PACK_SIZE (number of f64s consumed).
    pub fn unpack_atom(&mut self, buf: &[f64], is_ghost: bool) -> usize {
        self.tag.push(buf[0] as u32);
        self.origin_index.push(buf[1] as i32);
        self.skin.push(buf[2]);
        self.pos.push(Vector3::new(buf[3], buf[4], buf[5]));
        self.velocity.push(Vector3::new(buf[6], buf[7], buf[8]));
        self.quaterion.push(UnitQuaternion::from_quaternion(
            Quaternion::new(buf[9], buf[10], buf[11], buf[12])
        ));
        self.omega.push(Vector3::new(buf[13], buf[14], buf[15]));
        self.angular_momentum.push(Vector3::new(buf[16], buf[17], buf[18]));
        self.torque.push(Vector3::new(buf[19], buf[20], buf[21]));
        self.force.push(Vector3::new(buf[22], buf[23], buf[24]));
        self.mass.push(buf[25]);
        self.is_collision.push(buf[26] == 0.0);
        self.is_ghost.push(is_ghost);
        self.has_ghost.push(false);
        ATOM_PACK_SIZE
    }

    /// O(1) removal of atom i by swapping with the last element.
    pub fn swap_remove(&mut self, i: usize) {
        self.tag.swap_remove(i);
        self.origin_index.swap_remove(i);
        self.pos.swap_remove(i);
        self.velocity.swap_remove(i);
        self.quaterion.swap_remove(i);
        self.omega.swap_remove(i);
        self.angular_momentum.swap_remove(i);
        self.torque.swap_remove(i);
        self.force.swap_remove(i);
        self.skin.swap_remove(i);
        self.mass.swap_remove(i);
        self.is_collision.swap_remove(i);
        self.is_ghost.swap_remove(i);
        self.has_ghost.swap_remove(i);
    }

    /// Truncate all vecs to nlocal, removing ghost atoms in O(1).
    pub fn truncate_to_nlocal(&mut self) {
        let n = self.nlocal as usize;
        self.tag.truncate(n);
        self.origin_index.truncate(n);
        self.pos.truncate(n);
        self.velocity.truncate(n);
        self.quaterion.truncate(n);
        self.omega.truncate(n);
        self.angular_momentum.truncate(n);
        self.torque.truncate(n);
        self.force.truncate(n);
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
            .add_setup_system(read_input, ScheduleSetupSet::Setup)
            .add_update_system(remove_ghost_atoms, ScheduleSet::PostInitalIntegration)
            .add_update_system(zero_all_forces, ScheduleSet::PostInitalIntegration);
    }
}

pub fn read_input(
    input: Res<Input>,
    scheduler_manager: Res<SchedulerManager>,
    comm: Res<Comm>,
    domain: Res<Domain>,
    mut atom: ResMut<Atom>,
) {
    let commands = &input.current_commands[scheduler_manager.index];
    let mut rng = rand::rng();
    for c in commands.iter() {
        let values = c.split_whitespace().collect::<Vec<&str>>();

        if values.len() > 0 {
            match values[0] {
                "randomparticleinsert" => {
                    if comm.rank == 0 {
                        println!("Atom: {}", c);

                        let particles_to_add: u32 = values[1].parse::<u32>().unwrap();
                        let radius: f64 = values[2].parse::<f64>().unwrap();
                        let density: f64 = values[3].parse::<f64>().unwrap();
                        let mut max_tag = atom.get_max_tag();
                        let mut count = 0;

                        while count < particles_to_add {
                            let x = rng.random_range(domain.boundaries_low[0] + radius..domain.boundaries_high[0] - radius);
                            let y = rng.random_range(domain.boundaries_low[1] + radius..domain.boundaries_high[1] - radius);
                            let z = rng.random_range(domain.boundaries_low[2] + radius..domain.boundaries_high[2] - radius);

                            let pos = Vector3::<f64>::new(x, y, z);

                            let mut no_overlap = true;
                            for i in 0..atom.pos.len() {
                                let distance = (pos - atom.pos[i]).norm();
                                if distance <= (radius + radius) * 1.1 {
                                    no_overlap = false;
                                }
                            }

                            if no_overlap {
                                count += 1;
                                atom.natoms += 1;
                                atom.nlocal += 1;
                                atom.tag.push(max_tag);
                                atom.origin_index.push(0);
                                atom.skin.push(radius);
                                atom.is_collision.push(false);
                                atom.is_ghost.push(false);
                                atom.has_ghost.push(false);
                                max_tag += 1;
                                atom.pos.push(pos);
                                atom.velocity.push(Vector3::<f64>::zeros());
                                atom.quaterion.push(UnitQuaternion::identity());
                                atom.omega.push(Vector3::<f64>::zeros());
                                atom.angular_momentum.push(Vector3::<f64>::zeros());
                                atom.torque.push(Vector3::<f64>::zeros());
                                atom.force.push(Vector3::<f64>::zeros());
                                atom.mass.push(density * 4.0 / 3.0 * PI * radius.powi(3));
                            }
                        }
                    }
                }

                "randomparticlevelocity" => {
                    let rand_vel: f64 = values[1].parse::<f64>().unwrap();
                    for v in &mut atom.velocity {
                        let normal = Normal::new(0.0, rand_vel).unwrap();
                        v.x = normal.sample(&mut rand::rng());
                        v.y = normal.sample(&mut rand::rng());
                        v.z = normal.sample(&mut rand::rng());
                    }
                }

                _ => {}
            }
        }
    }
}

fn remove_ghost_atoms(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    atoms.truncate_to_nlocal();
    registry.truncate_all(atoms.nlocal as usize);
    atoms.nghost = 0;
}

fn zero_all_forces(mut atoms: ResMut<Atom>) {
    for i in 0..atoms.pos.len() {
        atoms.is_collision[i] = false;
        atoms.has_ghost[i] = false;
        atoms.force[i] = Vector3::zeros();
        atoms.torque[i] = Vector3::zeros();
    }
}

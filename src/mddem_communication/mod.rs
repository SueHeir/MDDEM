
use std::process::exit;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use mpi::traits::{Communicator, CommunicatorCollectives, Destination, Source};
use nalgebra::Vector3;
use crate::{mddem_atom::{Atom, AtomDataRegistry, ForceMPI}, mddem_domain::Domain, mddem_input::Input};

pub struct CommincationPlugin;

impl Plugin for CommincationPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Comm::new())
            .add_setup_system(read_input, ScheduleSetupSet::PreSetup)
            .add_setup_system(setup, ScheduleSetupSet::PostSetup)
            .add_update_system(exchange, ScheduleSet::Exchange)
            .add_update_system(borders, ScheduleSet::PreNeighbor)
            .add_update_system(reverse_send_force, ScheduleSet::PostForce);
    }
}

pub struct Comm {
    pub _universe: mpi::environment::Universe,
    pub world: mpi::topology::SimpleCommunicator,
    pub size: i32,
    pub rank: i32,

    pub processor_decomposition: Vector3<i32>,
    pub processor_position: Vector3<i32>,
    pub swap_directions: [Vector3<i32>; 2],
    pub periodic_swap: [Vector3<f64>; 2],
    pub send_amount: [Vector3<i32>; 2],
    pub recieve_amount: [Vector3<i32>; 2],
}

impl Comm {
    pub fn new() -> Self {
        let _universe: mpi::environment::Universe = mpi::initialize().unwrap();
        let world: mpi::topology::SimpleCommunicator = _universe.world();
        let size = world.size();
        let rank = world.rank();

        Comm {
            _universe,
            world,
            size,
            rank,
            processor_decomposition: Vector3::zeros(),
            processor_position: Vector3::zeros(),
            swap_directions: [Vector3::new(-1, -1, -1), Vector3::new(-1, -1, -1)],
            periodic_swap: [Vector3::zeros(), Vector3::zeros()],
            send_amount: [Vector3::new(0, 0, 0), Vector3::new(0, 0, 0)],
            recieve_amount: [Vector3::new(0, 0, 0), Vector3::new(0, 0, 0)],
        }
    }
}


pub fn read_input(input: Res<Input>, scheduler_manager: Res<SchedulerManager>, mut comm: ResMut<Comm>) {
    let commands = &input.current_commands[scheduler_manager.index];
    for c in commands.iter() {
        let values = c.split_whitespace().collect::<Vec<&str>>();

        if values.len() > 0 {
            match values[0] {
                "processors" => {
                    if comm.rank == 0 {
                        println!("Comm: {}", c);
                    }

                    comm.processor_decomposition.x = values[1].parse::<i32>().unwrap();
                    comm.processor_decomposition.y = values[2].parse::<i32>().unwrap();
                    comm.processor_decomposition.z = values[3].parse::<i32>().unwrap();

                    let mul = comm.processor_decomposition.x
                        * comm.processor_decomposition.y
                        * comm.processor_decomposition.z;
                    if mul != comm.size {
                        if comm.rank == 0 {
                            println!(
                                "Command: {0} with {1} processors does not match {2}",
                                c, mul, comm.size
                            );
                        }
                        exit(1);
                    }

                    let mut iter = 0;
                    for i in 0..comm.processor_decomposition[0] {
                        for j in 0..comm.processor_decomposition[1] {
                            for k in 0..comm.processor_decomposition[2] {
                                if iter == comm.rank {
                                    comm.processor_position = Vector3::new(i, j, k);
                                    println!("{:?}", comm.processor_position)
                                }
                                iter += 1;
                            }
                        }
                    }
                }

                _ => {}
            }
        }
    }
}

pub fn setup(mut comm: ResMut<Comm>, domain: Res<Domain>) {
    let mut iter = 0;
    for i in 0..comm.processor_decomposition[0] {
        for j in 0..comm.processor_decomposition[1] {
            for k in 0..comm.processor_decomposition[2] {
                if i == comm.processor_position.x + 1
                    && j == comm.processor_position.y
                    && k == comm.processor_position.z
                {
                    comm.swap_directions[1].x = iter
                }
                if i == comm.processor_position.x
                    && j == comm.processor_position.y + 1
                    && k == comm.processor_position.z
                {
                    comm.swap_directions[1].y = iter
                }
                if i == comm.processor_position.x
                    && j == comm.processor_position.y
                    && k == comm.processor_position.z + 1
                {
                    comm.swap_directions[1].z = iter
                }
                if comm.processor_position.x == comm.processor_decomposition.x - 1
                    && domain.is_periodic.x
                {
                    if i == 0 && j == comm.processor_position.y && k == comm.processor_position.z {
                        comm.swap_directions[1].x = iter;
                        comm.periodic_swap[1].x = -1.0;
                    }
                }
                if comm.processor_position.y == comm.processor_decomposition.y - 1
                    && domain.is_periodic.y
                {
                    if i == comm.processor_position.x && j == 0 && k == comm.processor_position.z {
                        comm.swap_directions[1].y = iter;
                        comm.periodic_swap[1].y = -1.0;
                    }
                }
                if comm.processor_position.z == comm.processor_decomposition.z - 1
                    && domain.is_periodic.z
                {
                    if i == comm.processor_position.x && j == comm.processor_position.y && k == 0 {
                        comm.swap_directions[1].z = iter;
                        comm.periodic_swap[1].z = -1.0;
                    }
                }
                if i == comm.processor_position.x - 1
                    && j == comm.processor_position.y
                    && k == comm.processor_position.z
                {
                    comm.swap_directions[0].x = iter
                }
                if i == comm.processor_position.x
                    && j == comm.processor_position.y - 1
                    && k == comm.processor_position.z
                {
                    comm.swap_directions[0].y = iter
                }
                if i == comm.processor_position.x
                    && j == comm.processor_position.y
                    && k == comm.processor_position.z - 1
                {
                    comm.swap_directions[0].z = iter
                }
                if comm.processor_position.x == 0 && domain.is_periodic.x {
                    if i == comm.processor_decomposition.x - 1
                        && j == comm.processor_position.y
                        && k == comm.processor_position.z
                    {
                        comm.swap_directions[0].x = iter;
                        comm.periodic_swap[0].x = 1.0;
                    }
                }
                if comm.processor_position.y == 0 && domain.is_periodic.y {
                    if i == comm.processor_position.x
                        && j == comm.processor_decomposition.y - 1
                        && k == comm.processor_position.z
                    {
                        comm.swap_directions[0].y = iter;
                        comm.periodic_swap[0].y = 1.0;
                    }
                }
                if comm.processor_position.z == 0 && domain.is_periodic.z {
                    if i == comm.processor_position.x
                        && j == comm.processor_position.y
                        && k == comm.processor_decomposition.z - 1
                    {
                        comm.swap_directions[0].z = iter;
                        comm.periodic_swap[0].z = 1.0;
                    }
                }
                iter += 1;
            }
        }
    }
}


// ── Helpers ───────────────────────────────────────────────────────────────────

/// Scan local atoms and pack those within the border skin for (dim, swap) into
/// `send_buff`. Returns the number of atoms packed. `periodic_offset` is the
/// domain-length shift applied to the position when wrapping periodically.
fn pack_border_atoms(
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    dim: usize,
    swap: usize,
    periodic_offset: f64,
    domain: &Domain,
    send_buff: &mut Vec<f64>,
    scan_end: usize,
) -> i32 {
    let mut count = 0i32;
    // Scan local atoms + any ghosts forwarded from earlier dimensions so that
    // atoms near a subdomain corner reach their diagonal neighbour's ghost list.
    for i in 0..scan_end {
        let in_skin = if swap == 0 {
            atoms.pos[i][dim] < domain.sub_domain_low[dim] + atoms.skin[i] * 4.0
        } else {
            atoms.pos[i][dim] >= domain.sub_domain_high[dim] - atoms.skin[i] * 4.0
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

/// Unpack `count` ghost atoms from a border receive buffer (has trailing count
/// sentinel). Increments `atoms.nghost` for each atom added.
fn unpack_ghost_atoms(
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    buf: &[f64],
    count: usize,
) {
    let data = &buf[..buf.len() - 1];
    let mut pos = 0;
    for _ in 0..count {
        pos += atoms.unpack_atom(&data[pos..], true);
        pos += registry.unpack_all(&data[pos..]);
        atoms.nghost += 1;
    }
}


// ── Exchange ──────────────────────────────────────────────────────────────────

pub fn exchange(
    comm: Res<Comm>,
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
) {
    comm.world.barrier();

    let mut atoms_buff: Vec<Vec<f64>> = (0..comm.size).map(|_| Vec::new()).collect();
    let mut counts = vec![0.0f64; comm.size as usize];

    for i in (0..atoms.pos.len()).rev() {
        let xi = (atoms.pos[i].x / domain.sub_length.x).floor() as i32;
        let yi = (atoms.pos[i].y / domain.sub_length.y).floor() as i32;
        let zi = (atoms.pos[i].z / domain.sub_length.z).floor() as i32;

        let to_processor = xi * comm.processor_decomposition.z * comm.processor_decomposition.y
            + yi * comm.processor_decomposition.z
            + zi;

        if to_processor != comm.rank {
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

    for p in 0..comm.size {
        if p == comm.rank {
            for _rec in 0..comm.size - 1 {
                let (msg, _status) = comm.world.any_process().receive_vec::<f64>();
                let msg_count = msg[msg.len() - 1] as usize;
                let data = &msg[..msg.len() - 1];
                let mut pos = 0;
                for _ in 0..msg_count {
                    pos += atoms.unpack_atom(&data[pos..], false);
                    pos += registry.unpack_all(&data[pos..]);
                }
            }
        } else {
            comm.world.process_at_rank(p).send(&atoms_buff[p as usize]);
        }
        comm.world.barrier();
    }

    let u = vec![atoms.pos.len() as i32; comm.size as usize];
    let mut v = vec![0i32; comm.size as usize];
    comm.world.all_to_all_into(&u, &mut v);
    comm.world.barrier();
}


// ── Borders ───────────────────────────────────────────────────────────────────

pub fn borders(
    mut comm: ResMut<Comm>,
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
) {
    let u = vec![atoms.pos.len() as i32; comm.size as usize];
    let mut v = vec![0i32; comm.size as usize];
    comm.world.all_to_all_into(&u, &mut v);
    atoms.natoms = v.iter().sum::<i32>() as u64;
    atoms.nlocal = atoms.pos.len() as u32;
    atoms.nghost = 0;

    comm.world.barrier();

    let mut send_buff: Vec<f64> = Vec::new();

    // scan_end grows after each dimension so that ghosts received in earlier
    // dimensions are forwarded to diagonal neighbours in later dimensions.
    // It does NOT grow within a dimension to avoid bouncing atoms back.
    let mut scan_end = atoms.nlocal as usize;

    for dim in 0..3 {
        let dim_scan_end = scan_end; // snapshot: constant for both swaps of this dim

        for swap in 0..2 {
            let to_proc   = comm.swap_directions[swap][dim];
            let from_proc = comm.swap_directions[(swap + 1) % 2][dim];
            let periodic_offset = comm.periodic_swap[swap][dim];

            comm.send_amount[swap][dim] = 0;
            comm.recieve_amount[swap][dim] = 0;
            send_buff.clear();

            if to_proc == from_proc && to_proc != comm.rank {

                if to_proc > comm.rank {
                    // Send first, receive second
                    if to_proc != -1 {
                        let count = pack_border_atoms(
                            &mut atoms, &registry, dim, swap, periodic_offset, &domain, &mut send_buff, dim_scan_end,
                        );
                        comm.send_amount[swap][dim] = count;
                        send_buff.push(count as f64);
                        comm.world.process_at_rank(to_proc).send(&send_buff);

                        let (msg, _status) = comm.world.process_at_rank(from_proc).receive_vec::<f64>();
                        let recv_count = msg[msg.len() - 1] as usize;
                        comm.recieve_amount[swap][dim] = recv_count as i32;
                        unpack_ghost_atoms(&mut atoms, &registry, &msg, recv_count);
                    }
                } else {
                    // Receive first, send second
                    let count = pack_border_atoms(
                        &mut atoms, &registry, dim, swap, periodic_offset, &domain, &mut send_buff, dim_scan_end,
                    );
                    comm.send_amount[swap][dim] = count;

                    if from_proc != -1 {
                        let (msg, _status) = comm.world.process_at_rank(to_proc).receive_vec::<f64>();
                        let recv_count = msg[msg.len() - 1] as usize;
                        comm.recieve_amount[swap][dim] = recv_count as i32;
                        unpack_ghost_atoms(&mut atoms, &registry, &msg, recv_count);

                        send_buff.push(count as f64);
                        comm.world.process_at_rank(to_proc).send(&send_buff);
                    }
                }

            } else {
                // Send and receive from different processors (or self-send)
                if to_proc != -1 {
                    let count = pack_border_atoms(
                        &mut atoms, &registry, dim, swap, periodic_offset, &domain, &mut send_buff, dim_scan_end,
                    );
                    comm.send_amount[swap][dim] = count;

                    if to_proc != comm.rank {
                        send_buff.push(count as f64);
                        comm.world.process_at_rank(to_proc).send(&send_buff);
                    } else {
                        // Self-send: write directly
                        comm.recieve_amount[swap][dim] = count;
                        let mut self_buf = send_buff.clone();
                        self_buf.push(count as f64);
                        unpack_ghost_atoms(&mut atoms, &registry, &self_buf, count as usize);
                    }
                }

                if from_proc != -1 && from_proc != comm.rank {
                    let (msg, _status) = comm.world.process_at_rank(from_proc).receive_vec::<f64>();
                    let recv_count = msg[msg.len() - 1] as usize;
                    comm.recieve_amount[swap][dim] = recv_count as i32;
                    unpack_ghost_atoms(&mut atoms, &registry, &msg, recv_count);
                }
            }

            comm.world.barrier();
        }

        // After both swaps of this dimension complete, include the newly received
        // ghosts when packing for the next dimension.
        scan_end = atoms.nlocal as usize + atoms.nghost as usize;
    }

    comm.world.barrier();

    let u = vec![atoms.nghost as i32; comm.size as usize];
    let mut v = vec![0i32; comm.size as usize];
    comm.world.all_to_all_into(&u, &mut v);
    comm.world.barrier();
}


// ── Reverse send force ────────────────────────────────────────────────────────

pub fn reverse_send_force(comm: Res<Comm>, mut atoms: ResMut<Atom>) {
    let mut send_buff: Vec<ForceMPI> = Vec::new();
    let mut recieve_position = atoms.pos.len() as i32;
    let mut _total_got: i32 = 0;

    for dim in (0..3).rev() {
        for swap in (0..2).rev() {
            let to_proc   = comm.swap_directions[swap][dim];
            let from_proc = comm.swap_directions[(swap + 1) % 2][dim];
            send_buff.clear();

            if to_proc == from_proc && to_proc != comm.rank {
                if from_proc < comm.rank {
                    if from_proc != -1 {
                        for i in (recieve_position - comm.recieve_amount[swap][dim])..recieve_position {
                            send_buff.push(atoms.get_force_data(i as usize));
                        }
                        comm.world.process_at_rank(from_proc).send(&send_buff);

                        let (msg, _status) = comm.world.process_at_rank(to_proc).receive_vec::<ForceMPI>();
                        for atom in msg {
                            _total_got += 1;
                            atoms.apply_force_data(atom, comm.rank, swap as i32, dim as i32);
                        }
                    }
                } else {
                    for i in (recieve_position - comm.recieve_amount[swap][dim])..recieve_position {
                        send_buff.push(atoms.get_force_data(i as usize));
                    }
                    if from_proc != -1 {
                        let (msg, _status) = comm.world.process_at_rank(to_proc).receive_vec::<ForceMPI>();
                        for atom in msg {
                            atoms.apply_force_data(atom, comm.rank, swap as i32, dim as i32);
                            _total_got += 1;
                        }
                        comm.world.process_at_rank(from_proc).send(&send_buff);
                    }
                }
            } else {
                if from_proc != -1 {
                    for i in (recieve_position - comm.recieve_amount[swap][dim])..recieve_position {
                        send_buff.push(atoms.get_force_data(i as usize));
                    }
                    if from_proc != comm.rank {
                        comm.world.process_at_rank(from_proc).send(&send_buff);
                    } else {
                        let msg = send_buff.clone();
                        for atom in msg {
                            _total_got += 1;
                            atoms.apply_force_data(atom, comm.rank, swap as i32, dim as i32);
                        }
                    }
                }
                if to_proc != -1 && to_proc != comm.rank {
                    let (msg, _status) = comm.world.process_at_rank(to_proc).receive_vec::<ForceMPI>();
                    for atom in msg {
                        _total_got += 1;
                        atoms.apply_force_data(atom, comm.rank, swap as i32, dim as i32);
                    }
                }
            }

            recieve_position -= comm.recieve_amount[swap][dim];
            comm.world.barrier();
        }
    }

    comm.world.barrier();

    let u = vec![atoms.nghost as i32; comm.size as usize];
    let mut v = vec![0i32; comm.size as usize];
    comm.world.all_to_all_into(&u, &mut v);
    comm.world.barrier();
}

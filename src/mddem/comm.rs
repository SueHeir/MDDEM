use mpi::traits::*;
use nalgebra::Vector3;
use std::process::exit;

use super::{
    atom::{Atom, AtomMPI, ForceMPI},
    domain::Domain,
    input::Input,
    scheduler::{Res, ResMut, ScheduleSet::*, Scheduler},
};

pub fn comm_app(scheduler: &mut Scheduler) {
    scheduler.add_resource(Comm::new());
    scheduler.add_setup_system(read_input, Setup);
    scheduler.add_setup_system(setup, PreNeighbor);

    scheduler.add_update_system(exchange, Exchange);
    scheduler.add_update_system(borders, PreNeighbor);
    scheduler.add_update_system(reverse_send_force, PostForce);
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
            send_amount: [Vector3::new(0, 0, 0), Vector3::new(0,0,0)],
            recieve_amount: [Vector3::new(0,0,0), Vector3::new(0,0,0)],

        }
    }
}

pub fn read_input(input: Res<Input>, mut comm: ResMut<Comm>) {
    let commands = &input.commands;
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
                                //You're Processor
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
                //Up
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
                //Perodic up
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

                //Down
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

                //Perodic Down
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

    // println!("{}  positive: {:?} negative: {:?}",comm.rank, comm.swap_directions[1], comm.swap_directions[0]);
}

pub fn exchange(comm: Res<Comm>, mut atoms: ResMut<Atom>, domain: Res<Domain>) {

    comm.world.barrier();
    //Collect Atoms outside of domain
    let mut atoms_mpi: Vec<Vec<AtomMPI>> = Vec::new();

    for _p in 0..comm.size {
        atoms_mpi.push(Vec::new());
    }

    for i in (0..atoms.pos.len()).rev() {
        let xi = (atoms.pos[i].x / domain.sub_length.x).floor() as i32;
        let yi = (atoms.pos[i].y / domain.sub_length.y).floor() as i32;
        let zi = (atoms.pos[i].z / domain.sub_length.z).floor() as i32;

        let to_processor = xi * comm.processor_decomposition.z * comm.processor_decomposition.y
            + yi * comm.processor_decomposition.z
            + zi;
        // let to_processor = xi + yi * comm.processor_decomposition.x +
        //     zi * comm.processor_decomposition.y * comm.processor_decomposition.x;

        // if atoms.pos[i].x >= domain.sub_domain_high.x || atoms.pos[i].x < domain.sub_domain_low.x ||
        //     atoms.pos[i].y >= domain.sub_domain_high.y || atoms.pos[i].y < domain.sub_domain_low.y ||
        //     atoms.pos[i].z >= domain.sub_domain_high.z || atoms.pos[i].z < domain.sub_domain_low.z {
        //     println!("xi {} yi {}", xi, yi);
        // }
       
        if to_processor != comm.rank {
            
            let value = atoms.get_atom_mpi(i);
            // println!("rank: {} to processor: {}", comm.rank, to_processor);
            atoms_mpi[to_processor as usize].push(value)
        }
    }

    for p in 0..comm.size {
        //recieve atoms
        if p == comm.rank {
            for _rec in 0..comm.size - 1 {
                let (msg, _status) = comm.world.any_process().receive_vec::<AtomMPI>();

                // if msg.len() > 0 {
                //     println!("recieve {:?}", msg);
                // }
                for atom in msg {
                    let xi = (atom.pos.x / domain.sub_length.x).floor() as i32;
                    let yi = (atom.pos.y / domain.sub_length.y).floor() as i32;
                    let zi = (atom.pos.z / domain.sub_length.z).floor() as i32;

                    

                    let to_processor = xi * comm.processor_decomposition.z * comm.processor_decomposition.y
                    + yi * comm.processor_decomposition.z
                    + zi;
                    // let to_processor = xi + yi * comm.processor_decomposition.x +
                    //     zi * comm.processor_decomposition.y * comm.processor_decomposition.x;

                    if to_processor == comm.rank {
                        // println!("add atoms {:?}", atom);
                        atoms.add_atom_from_atom_mpi(atom, false);
                    } else {
                        println!("Exchanged atom which did not belong to new processor")
                    }
                }
            }
        }
        //send atoms
        else {
            //send buff to processor p
            comm.world.process_at_rank(p).send(&atoms_mpi[p as usize]);
            // if atoms_mpi[p as usize].len() > 0 {
            //     println!("send {:?}", atoms_mpi[p as usize]);
            // }
        }
        comm.world.barrier();
    }

    let u = vec![atoms.radius.len() as i32; comm.size as usize];
    let mut v = vec![0; comm.size as usize];

    comm.world.all_to_all_into(&u[..], &mut v[..]);

    comm.world.barrier();

    // if v.iter().sum::<i32>() > 0 {
    //     println!("Rank: {} Total real: {} Each Proc:{:?}",comm.rank, v.iter().sum::<i32>(), v);
    // }
}

pub fn borders(mut comm: ResMut<Comm>, mut atoms: ResMut<Atom>, domain: Res<Domain>) {
    let mut send_buff: Vec<AtomMPI> = Vec::new();

    let u = vec![atoms.radius.len() as i32; comm.size as usize];
    let mut v = vec![0; comm.size as usize];
    comm.world.all_to_all_into(&u[..], &mut v[..]);
    atoms.natoms = v.iter().sum::<i32>() as u64;
    atoms.nlocal = atoms.radius.len() as u32;
    atoms.nghost = 0;

    comm.world.barrier();

   

    for dim in 0..3 {
        for swap in 0..2 {
            let to_proc = comm.swap_directions[swap][dim];
            let from_proc = comm.swap_directions[(swap + 1) % 2][dim];

            comm.send_amount[swap][dim] = 0;
            comm.recieve_amount[swap][dim] = 0;
            send_buff.clear();
            // Avoid deadlocking for sending and recieving same processor
            if to_proc == from_proc && to_proc != comm.rank {

                //Send First, Recieve Second
                if to_proc > comm.rank {
                    //Send
                    if to_proc != -1 {
                        for i in 0..atoms.radius.len() {
                            if swap == 0 {
                                if atoms.pos[i][dim] < domain.sub_domain_low[dim] + atoms.radius[i] {
                                    let mut atom = atoms.copy_atom_mpi(i);
                                    atom.pos[&dim] += comm.periodic_swap[swap][dim] * domain.size[dim];
                                    send_buff.push(atom);
                                    comm.send_amount[swap][dim] +=1;
                                }
                            } else {
                                if atoms.pos[i][dim] >= domain.sub_domain_high[dim] - atoms.radius[i] {
                                    let mut atom = atoms.copy_atom_mpi(i);
                                    atom.pos[&dim] += comm.periodic_swap[swap][dim] * domain.size[dim];
                                    send_buff.push(atom);
                                    comm.send_amount[swap][dim] +=1;
                                }
                            }
                        }
                        
                        comm.world.process_at_rank(to_proc).send(&send_buff);
                    
                        //Receive
                        // println!(
                        //     "{} from proc {} ",
                        //     comm.rank,
                        //     from_proc
                        // );
                        
                        let (msg, _status) = comm
                            .world
                            .process_at_rank(from_proc)
                            .receive_vec::<AtomMPI>();

                        // println!(
                        //     "{} recv from {} on dim {}",
                        //     comm.rank,
                        //     status.source_rank(),
                        //     dim
                        // );
                        comm.recieve_amount[swap][dim] = msg.len() as i32;
                        for atom in msg {
                            atoms.nghost += 1;
                            atoms.add_atom_from_atom_mpi(atom, true);
                        }
                    }
                    
                //Send Second, Recieve First
                } else {

                    for i in 0..atoms.radius.len() {
                        if swap == 0 {
                            if atoms.pos[i][dim] < domain.sub_domain_low[dim] + atoms.radius[i] {
                                let mut atom = atoms.copy_atom_mpi(i);
                                atom.pos[&dim] += comm.periodic_swap[swap][dim] * domain.size[dim];
                                send_buff.push(atom);
                                comm.send_amount[swap][dim] +=1;
                            }
                        } else {
                            if atoms.pos[i][dim] >= domain.sub_domain_high[dim] - atoms.radius[i] {
                                let mut atom = atoms.copy_atom_mpi(i);
                                atom.pos[&dim] += comm.periodic_swap[swap][dim] * domain.size[dim];
                                send_buff.push(atom);
                                comm.send_amount[swap][dim] +=1;
                            }
                        }
                    }
                    //Receive
                    // println!(
                    //     "{} from proc {} ",
                    //     comm.rank,
                    //     from_proc
                    // );
                    if from_proc != -1 {
                        let (msg, _status) = comm
                            .world
                            .process_at_rank(from_proc)
                            .receive_vec::<AtomMPI>();

                        // println!(
                        //     "{} recv from {} on dim {}",
                        //     comm.rank,
                        //     status.source_rank(),
                        //     dim
                        // );

                        comm.recieve_amount[swap][dim] = msg.len() as i32;
                        for atom in msg {
                            atoms.nghost += 1;
                            atoms.add_atom_from_atom_mpi(atom, true);
                        }
                    
                        //Send
                        
                        // println!("{} sent to {} on dim {}", comm.rank, to_proc, dim);
                        comm.world.process_at_rank(to_proc).send(&send_buff);
                    }
                }
            //Sending And Recieveing from different processors
            } else {
                //Send
                if to_proc != -1 {
                    for i in 0..atoms.radius.len() {
                        if swap == 0 {
                            if atoms.pos[i][dim] < domain.sub_domain_low[dim] + atoms.radius[i] {
                                let mut atom = atoms.copy_atom_mpi(i);
                                atom.pos[&dim] += comm.periodic_swap[swap][dim] * domain.size[dim];
                                send_buff.push(atom);
                                comm.send_amount[swap][dim] +=1;
                            }
                        } else {
                            if atoms.pos[i][dim] >= domain.sub_domain_high[dim] - atoms.radius[i] {
                                let mut atom = atoms.copy_atom_mpi(i);
                                atom.pos[&dim] += comm.periodic_swap[swap][dim] * domain.size[dim];
                                send_buff.push(atom);
                                comm.send_amount[swap][dim] +=1;
                            }
                        }
                    }
                    if to_proc != comm.rank {
                        // println!("{} sent to {} on dim {}", comm.rank, to_proc, dim);
                        comm.world.process_at_rank(to_proc).send(&send_buff);
                    } else {
                        let msg = send_buff.clone();
                        comm.recieve_amount[swap][dim] = msg.len() as i32;
                        for atom in msg {
                            atoms.nghost += 1;
                            atoms.add_atom_from_atom_mpi(atom, true);
                        }
                    }
                }
                //Receive
                // println!(
                //     "{} from proc {} ",
                //     comm.rank,
                //     from_proc
                // );
                if from_proc != -1 && from_proc != comm.rank {
                    let (msg, _status) = comm
                        .world
                        .process_at_rank(from_proc)
                        .receive_vec::<AtomMPI>();

                    // println!(
                    //     "{} recv from {} on dim {}",
                    //     comm.rank,
                    //     status.source_rank(),
                    //     dim
                    // );
                    comm.recieve_amount[swap][dim] = msg.len() as i32;
                    for atom in msg {
                        atoms.nghost += 1;
                        atoms.add_atom_from_atom_mpi(atom, true);
                    }
                }
            }
           

            comm.world.barrier();
        }
    }
    comm.world.barrier();

    let u = vec![atoms.nghost as i32; comm.size as usize];
    let mut v = vec![0; comm.size as usize];
    comm.world.all_to_all_into(&u[..], &mut v[..]);
    // println!("total ghost atoms: {}", v.iter().sum::<i32>());

    comm.world.barrier();
}




pub fn reverse_send_force(mut comm: ResMut<Comm>, mut atoms: ResMut<Atom>) {
    let mut send_buff: Vec<ForceMPI> = Vec::new();

    let mut send_position = atoms.radius.len() as i32;
    let mut recieve_position = atoms.radius.len() as i32;

    // println!("{:?}",comm.recieve_amount);

    // println!("ghosts {} {}", atoms.nghost, (comm.recieve_amount[0].iter().sum::<i32>() + comm.recieve_amount[1].iter().sum::<i32>()));
    let mut total_got = 0;
   
    
    // println!("{:?}", atoms.origin_index);
    for dim in (0..3).rev() {
        for swap in (0..2).rev() {
            let to_proc = comm.swap_directions[swap][dim];
            let from_proc = comm.swap_directions[(swap + 1) % 2][dim];
            send_buff.clear();

            // println!("this swap {} dim {} total {}", swap, dim, comm.send_amount[swap][dim]);
            // Avoid deadlocking for sending and recieving same processor
            if to_proc == from_proc && to_proc != comm.rank {

                //Send First, Recieve Second
                if from_proc < comm.rank {
                    //Send
                    if from_proc != -1 {
                        for i in ((recieve_position - comm.recieve_amount[swap][dim])..recieve_position) {
                           send_buff.push(atoms.get_force_data(i as usize));
                        }
                        
                        comm.world.process_at_rank(from_proc).send(&send_buff);
                    
                        
                        let (msg, _status) = comm
                            .world
                            .process_at_rank(to_proc)
                            .receive_vec::<ForceMPI>();


                        for atom in msg {
                            total_got += 1;
                            atoms.apply_force_data(atom, comm.rank, swap as i32, dim as i32);
                        }
                    }
                    
                //Send Second, Recieve First
                } else {

                    for i in ((recieve_position - comm.recieve_amount[swap][dim])..recieve_position) {
                        send_buff.push(atoms.get_force_data(i as usize));
                    }

                    if from_proc != -1 {
                        let (msg, _status) = comm
                            .world
                            .process_at_rank(to_proc)
                            .receive_vec::<ForceMPI>();

                        

                        for atom in msg {
                            atoms.apply_force_data(atom, comm.rank, swap as i32, dim as i32);
                            total_got += 1;
                        }
                    
                        //Send
                        
                        comm.world.process_at_rank(from_proc).send(&send_buff);
                    }
                }
            //Sending And Recieveing from different processors
            } else {
                //Send
                if from_proc != -1 {
                    for i in ((recieve_position - comm.recieve_amount[swap][dim])..recieve_position) {
                        send_buff.push(atoms.get_force_data(i as usize));
                    }
                    if from_proc != comm.rank {
                        comm.world.process_at_rank(from_proc).send(&send_buff);
                    } else {
                        let msg = send_buff.clone();
                        for atom in msg {
                            total_got += 1;
                            atoms.apply_force_data(atom, comm.rank, swap as i32, dim as i32);
                        }
                    }
                }
                if to_proc != -1 && to_proc != comm.rank {
                    let (msg, _status) = comm
                        .world
                        .process_at_rank(to_proc)
                        .receive_vec::<ForceMPI>();
                    for atom in msg {
                        total_got += 1;
                        atoms.apply_force_data(atom, comm.rank, swap as i32, dim as i32);
                    }
                }
            }
            // println!("next swap {} dim {}", swap, dim);
            recieve_position -= comm.recieve_amount[swap][dim];
            
            comm.world.barrier();
        }
    }

    // println!("total_got {} {}", total_got, (comm.send_amount[0].iter().sum::<i32>() + comm.send_amount[1].iter().sum::<i32>()));

    
    comm.world.barrier();

    let u = vec![atoms.nghost as i32; comm.size as usize];
    let mut v = vec![0; comm.size as usize];
    comm.world.all_to_all_into(&u[..], &mut v[..]);
    // println!("total ghost atoms: {}", v.iter().sum::<i32>());

    comm.world.barrier();
}





use std::process::exit;

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use mpi::traits::{Communicator, CommunicatorCollectives, Destination, Source};
use nalgebra::Vector3;
use crate::{mddem_atom::{Atom, AtomAdded, AtomMPI, ForceMPI}, mddem_domain::Domain, mddem_input::Input};

pub struct CommincationPlugin;

impl Plugin for CommincationPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Comm::new())
            .add_setup_system(read_input, ScheduleSet::Setup)
            .add_setup_system(setup, ScheduleSet::PreNeighbor)
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
            send_amount: [Vector3::new(0, 0, 0), Vector3::new(0,0,0)],
            recieve_amount: [Vector3::new(0,0,0), Vector3::new(0,0,0)],

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
}

pub fn exchange(comm: Res<Comm>, mut atoms: ResMut<Atom>, domain: Res<Domain>) {
    // println!("exchange");
    comm.world.barrier();
    //Collect Atoms outside of domain
    // let mut atoms_mpi: Vec<Vec<AtomMPI>> = Vec::new();
    // let mut atoms_added_mpi: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut atoms_buff: Vec<Vec<f64>> = Vec::new();
    let mut counts = Vec::new();

    for _p in 0..comm.size {
        atoms_buff.push(Vec::new());
        counts.push(0.0);
    }

    for i in (0..atoms.pos.len()).rev() {
        // println!("{:?}", atoms.pos[i]);
        let xi = (atoms.pos[i].x / domain.sub_length.x).floor() as i32;
        let yi = (atoms.pos[i].y / domain.sub_length.y).floor() as i32;
        let zi = (atoms.pos[i].z / domain.sub_length.z).floor() as i32;

       
        let to_processor = xi * comm.processor_decomposition.z * comm.processor_decomposition.y
            + yi * comm.processor_decomposition.z
            + zi;

        if to_processor != comm.rank {
           
            counts[to_processor as usize] += 1.0;
            atoms_buff[to_processor as usize].append(&mut atoms.get_atom_buff(i));

        
            for (_type_id, ref_cell) in &atoms.added {
                let mut atom_added_binder = ref_cell.borrow_mut();
                let atom_added = atom_added_binder.as_mut();
                
                atoms_buff[to_processor as usize].append(&mut atom_added.get_mpi(i));
            }
        }
    }

    for (buff, count) in atoms_buff.iter_mut().zip(counts) {
        buff.push(count);
    }
   


    for p in 0..comm.size {
        //recieve atoms
        if p == comm.rank {
            for _rec in 0..comm.size - 1 {
                let (mut msg, _status) = comm.world.any_process().receive_vec::<f64>();

                let msg_count = msg[msg.len()-1] as usize;

                for _i in 0..msg_count {
                    msg = atoms.add_atom_from_buff(msg, false);

                    for (_type_id, ref_cell) in &atoms.added {
                        let mut atom_added_binder = ref_cell.borrow_mut();
                        let atom_added = atom_added_binder.as_mut();
                        
                        
                        msg = atom_added.set_mpi(msg);
                      
                    }
                }

                if msg.len() != 1 {
                    println!("exchange {}", msg.len());
                }
                

            }
        }
        //send atoms
        else {
            //send buff to processor p
            comm.world.process_at_rank(p).send(&atoms_buff[p as usize]);
        }
        comm.world.barrier();
    }

    let u = vec![atoms.pos.len() as i32; comm.size as usize];
    let mut v = vec![0; comm.size as usize];

    comm.world.all_to_all_into(&u[..], &mut v[..]);

    comm.world.barrier();

    // if v.iter().sum::<i32>() > 0 {
    //     println!("Rank: {} Total real: {} Each Proc:{:?}",comm.rank, v.iter().sum::<i32>(), v);
    // }
}


pub fn borders(mut comm: ResMut<Comm>, mut atoms: ResMut<Atom>, domain: Res<Domain>) {
    // println!("borders");

    let mut send_buff: Vec<f64> = Vec::new();

    let u = vec![atoms.pos.len() as i32; comm.size as usize];
    let mut v = vec![0; comm.size as usize];
    comm.world.all_to_all_into(&u[..], &mut v[..]);
    atoms.natoms = v.iter().sum::<i32>() as u64;
    atoms.nlocal = atoms.pos.len() as u32;
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
                        for i in 0..atoms.pos.len() {
                            if swap == 0 {
                                // println!("{} {} {}", atoms.pos.len(),domain.sub_domain_low.len(),atoms.skin.len());
                                if atoms.pos[i][dim] < domain.sub_domain_low[dim] + atoms.skin[i] {
                                    let mut change_pos = Vector3::new(0.0,0.0,0.0);
                                    change_pos[dim] += comm.periodic_swap[swap][dim] * domain.size[dim];
                                    send_buff.append(&mut atoms.clone_atom_buff(i, change_pos));
                                    comm.send_amount[swap][dim] +=1;
                                    for (_type_id, ref_cell) in &atoms.added {
                                        let mut atom_added_binder = ref_cell.borrow_mut();
                                        let atom_added = atom_added_binder.as_mut();
                                        
                                       send_buff.append(&mut atom_added.copy_mpi(i));
                                    }

                                }
                            } else {
                                if atoms.pos[i][dim] >= domain.sub_domain_high[dim] - atoms.skin[i] {
                                    let mut change_pos = Vector3::new(0.0,0.0,0.0);
                                    change_pos[dim] += comm.periodic_swap[swap][dim] * domain.size[dim];
                                    send_buff.append(&mut atoms.clone_atom_buff(i, change_pos));
                                    comm.send_amount[swap][dim] +=1;
                                    for (_type_id, ref_cell) in &atoms.added {
                                        let mut atom_added_binder = ref_cell.borrow_mut();
                                        let atom_added = atom_added_binder.as_mut();
                                        
                                       send_buff.append(&mut atom_added.copy_mpi(i));
                                    }
                                }
                            }
                        }

                        send_buff.push(comm.send_amount[swap][dim] as f64);
                        comm.world.process_at_rank(to_proc).send(&send_buff);
                    
                        //Receive
                        // println!(
                        //     "{} from proc {} ",
                        //     comm.rank,
                        //     from_proc
                        // );
                        
                        let (mut msg, _status) = comm
                            .world
                            .process_at_rank(from_proc)
                            .receive_vec::<f64>();

                        // println!(
                        //     "{} recv from {} on dim {}",
                        //     comm.rank,
                        //     status.source_rank(),
                        //     dim
                        // );
                        comm.recieve_amount[swap][dim] = msg[msg.len()-1] as i32;
                        for _i in 0..comm.recieve_amount[swap][dim] {
                            msg = atoms.add_atom_from_buff(msg, true);
                            for (_type_id, ref_cell) in &atoms.added {
                                let mut atom_added_binder = ref_cell.borrow_mut();
                                let atom_added = atom_added_binder.as_mut();
                                
                                msg = atom_added.set_mpi(msg);
                            }
                            atoms.nghost += 1;
                            // atoms.add_atom_from_atom_mpi(atom, true);
                        }
                        if msg.len() != 1 {
                            println!("borders {}", msg.len());
                        }
                    }
                    
                //Send Second, Recieve First
                } else {

                    for i in 0..atoms.pos.len() {
                        if swap == 0 {
                            if atoms.pos[i][dim] < domain.sub_domain_low[dim] + atoms.skin[i] {
                                let mut change_pos = Vector3::new(0.0,0.0,0.0);
                                change_pos[dim] += comm.periodic_swap[swap][dim] * domain.size[dim];
                                send_buff.append(&mut atoms.clone_atom_buff(i, change_pos));
                                comm.send_amount[swap][dim] +=1;
                                for (_type_id, ref_cell) in &atoms.added {
                                    let mut atom_added_binder = ref_cell.borrow_mut();
                                    let atom_added = atom_added_binder.as_mut();
                                    
                                    send_buff.append(&mut atom_added.copy_mpi(i));
                                }
                            }
                        } else {
                            if atoms.pos[i][dim] >= domain.sub_domain_high[dim] - atoms.skin[i] {
                                let mut change_pos = Vector3::new(0.0,0.0,0.0);
                                change_pos[dim] += comm.periodic_swap[swap][dim] * domain.size[dim];
                                send_buff.append(&mut atoms.clone_atom_buff(i, change_pos));
                                comm.send_amount[swap][dim] +=1;
                                for (_type_id, ref_cell) in &atoms.added {
                                    let mut atom_added_binder = ref_cell.borrow_mut();
                                    let atom_added = atom_added_binder.as_mut();
                                    
                                    send_buff.append(&mut atom_added.copy_mpi(i));
                                }
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
                        let (mut msg, _status) = comm
                            .world
                            .process_at_rank(from_proc)
                            .receive_vec::<f64>();

                        // println!(
                        //     "{} recv from {} on dim {}",
                        //     comm.rank,
                        //     status.source_rank(),
                        //     dim
                        // );

                        comm.recieve_amount[swap][dim] = msg[msg.len()-1] as i32;
                        for _i in 0..comm.recieve_amount[swap][dim] {
                            msg = atoms.add_atom_from_buff(msg, true);
                            for (_type_id, ref_cell) in &atoms.added {
                                let mut atom_added_binder = ref_cell.borrow_mut();
                                let atom_added = atom_added_binder.as_mut();
                                
                                msg = atom_added.set_mpi(msg);
                            }
                            atoms.nghost += 1;
                            // atoms.add_atom_from_atom_mpi(atom, true);
                        }
                        if msg.len() != 1 {
                            println!("borders {}", msg.len());
                        }
                    
                        //Send
                        
                        // println!("{} sent to {} on dim {}", comm.rank, to_proc, dim);
                        send_buff.push(comm.send_amount[swap][dim] as f64);
                        comm.world.process_at_rank(to_proc).send(&send_buff);
                    }
                }
            //Sending And Recieveing from different processors
            } else {
                //Send
                if to_proc != -1 {
                    for i in 0..atoms.pos.len() {
                        if swap == 0 {
                            if atoms.pos[i][dim] < domain.sub_domain_low[dim] + atoms.skin[i] {
                                let mut change_pos = Vector3::new(0.0,0.0,0.0);
                                change_pos[dim] += comm.periodic_swap[swap][dim] * domain.size[dim];
                                send_buff.append(&mut atoms.clone_atom_buff(i, change_pos));
                                comm.send_amount[swap][dim] +=1;
                                for (_type_id, ref_cell) in &atoms.added {
                                    let mut atom_added_binder = ref_cell.borrow_mut();
                                    let atom_added = atom_added_binder.as_mut();
                                    
                                   send_buff.append(&mut atom_added.copy_mpi(i));
                                }
                            }
                        } else {
                            if atoms.pos[i][dim] >= domain.sub_domain_high[dim] - atoms.skin[i] {
                                let mut change_pos = Vector3::new(0.0,0.0,0.0);
                                change_pos[dim] += comm.periodic_swap[swap][dim] * domain.size[dim];
                                send_buff.append(&mut atoms.clone_atom_buff(i, change_pos));
                                comm.send_amount[swap][dim] +=1;
                                for (_type_id, ref_cell) in &atoms.added {
                                    let mut atom_added_binder = ref_cell.borrow_mut();
                                    let atom_added = atom_added_binder.as_mut();
                                    
                                    send_buff.append(&mut atom_added.copy_mpi(i));
                                }
                            }
                        }
                    }
                    if to_proc != comm.rank {
                        // println!("{} sent to {} on dim {}", comm.rank, to_proc, dim);
                        send_buff.push(comm.send_amount[swap][dim] as f64);
                        comm.world.process_at_rank(to_proc).send(&send_buff);
                    } else {
                        let mut msg = send_buff.clone();
                        comm.recieve_amount[swap][dim] = comm.send_amount[swap][dim];
                        for _i in 0..comm.recieve_amount[swap][dim] {
                            msg = atoms.add_atom_from_buff(msg, true);
                            for (_type_id, ref_cell) in &atoms.added {
                                let mut atom_added_binder = ref_cell.borrow_mut();
                                let atom_added = atom_added_binder.as_mut();
                                
                                msg = atom_added.set_mpi(msg);
                            }
                            atoms.nghost += 1;
                            // atoms.add_atom_from_atom_mpi(atom, true);
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
                    let (mut msg, _status) = comm
                        .world
                        .process_at_rank(from_proc)
                        .receive_vec::<f64>();

                    // println!(
                    //     "{} recv from {} on dim {}",
                    //     comm.rank,
                    //     status.source_rank(),
                    //     dim
                    // );
                    comm.recieve_amount[swap][dim] = msg[msg.len()-1] as i32;
                    for _i in 0..comm.recieve_amount[swap][dim] {
                        msg = atoms.add_atom_from_buff(msg, true);
                        for (_type_id, ref_cell) in &atoms.added {
                            let mut atom_added_binder = ref_cell.borrow_mut();
                            let atom_added = atom_added_binder.as_mut();
                            
                            msg = atom_added.set_mpi(msg);
                        }
                        atoms.nghost += 1;
                        // atoms.add_atom_from_atom_mpi(atom, true);
                    }
                    if msg.len() != 1 {
                        println!("borders {}", msg.len());
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




pub fn reverse_send_force(comm: Res<Comm>, mut atoms: ResMut<Atom>) {

    // println!("reverse send");

    let mut send_buff: Vec<ForceMPI> = Vec::new();

    let mut recieve_position = atoms.pos.len() as i32;

    // println!("{:?}",comm.recieve_amount);

    // println!("ghosts {} {}", atoms.nghost, (comm.recieve_amount[0].iter().sum::<i32>() + comm.recieve_amount[1].iter().sum::<i32>()));
    let mut _total_got: i32 = 0;
   
    
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
                        for i in (recieve_position - comm.recieve_amount[swap][dim])..recieve_position {
                           send_buff.push(atoms.get_force_data(i as usize));
                        }
                        
                        comm.world.process_at_rank(from_proc).send(&send_buff);
                    
                        
                        let (msg, _status) = comm
                            .world
                            .process_at_rank(to_proc)
                            .receive_vec::<ForceMPI>();


                        for atom in msg {
                            _total_got += 1;
                            atoms.apply_force_data(atom, comm.rank, swap as i32, dim as i32);
                        }
                    }
                    
                //Send Second, Recieve First
                } else {

                    for i in (recieve_position - comm.recieve_amount[swap][dim])..recieve_position {
                        send_buff.push(atoms.get_force_data(i as usize));
                    }

                    if from_proc != -1 {
                        let (msg, _status) = comm
                            .world
                            .process_at_rank(to_proc)
                            .receive_vec::<ForceMPI>();

                        

                        for atom in msg {
                            atoms.apply_force_data(atom, comm.rank, swap as i32, dim as i32);
                            _total_got += 1;
                        }
                    
                        //Send
                        
                        comm.world.process_at_rank(from_proc).send(&send_buff);
                    }
                }
            //Sending And Recieveing from different processors
            } else {
                //Send
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
                    let (msg, _status) = comm
                        .world
                        .process_at_rank(to_proc)
                        .receive_vec::<ForceMPI>();
                    for atom in msg {
                        _total_got += 1;
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


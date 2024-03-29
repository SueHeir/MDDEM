use std::process::exit;

use mpi::request::WaitGuard;
use mpi::traits::*;
use nalgebra::Vector3;

pub struct Comm {
    pub universe: mpi::environment::Universe,
    pub world: mpi::topology::SimpleCommunicator,
    pub size: i32,
    pub rank: i32,

    pub processor_decomposition: Vector3<i32>,
}

impl Comm {
    pub fn new() -> Self {
        let universe: mpi::environment::Universe = mpi::initialize().unwrap();
        let world: mpi::topology::SimpleCommunicator = universe.world();
        let size = world.size();
        let rank = world.rank();

        Comm {
            universe,
            world,
            size,
            rank,
            processor_decomposition: Vector3::zeros(),
        }
    }

    pub fn input(&mut self, commands: &Vec<String>) {
        for c in commands.iter() {
            let values = c.split_whitespace().collect::<Vec<&str>>();

            if values.len() > 0 {
                match values[0] {
                    "processors" => {
                        if self.rank == 0 {
                            println!("Comm: {}", c);
                        }

                        self.processor_decomposition.x = values[1].parse::<i32>().unwrap();
                        self.processor_decomposition.y = values[2].parse::<i32>().unwrap();
                        self.processor_decomposition.z = values[3].parse::<i32>().unwrap();

                        let mul = self.processor_decomposition.x
                            * self.processor_decomposition.y
                            * self.processor_decomposition.z;
                        if mul != self.size {
                            if self.rank == 0 {
                                println!(
                                    "Command: {0} with {1} processors does not match {2}",
                                    c, mul, self.size
                                );
                            }
                            exit(1);
                        }
                    }

                    _ => {}
                }
            }
        }

        // let next_rank = (rank + 1) % size;
        // let previous_rank = (rank - 1 + size) % size;

        // let msg = vec![rank, 2 * rank, 4 * rank];
        // mpi::request::scope(|scope| {
        //     let _sreq = WaitGuard::from(
        //         world
        //             .process_at_rank(next_rank)
        //             .immediate_send(scope, &msg[..]),
        //     );

        //     let (msg, status) = world.any_process().receive_vec();

        //     println!(
        //         "Process {} got message {:?}.\nStatus is: {:?}",
        //         rank, msg, status
        //     );
        //     let x = status.source_rank();
        //     assert_eq!(x, previous_rank);
        //     assert_eq!(vec![x, 2 * x, 4 * x], msg);

        //     let root_rank = 0;
        //     let root_process = world.process_at_rank(root_rank);

        //     let mut a;
        //     if world.rank() == root_rank {
        //         a = vec![2, 4, 8, 16];
        //         println!("Root broadcasting value: {:?}.", &a[..]);
        //     } else {
        //         a = vec![0; 4];
        //     }
        //     root_process.broadcast_into(&mut a[..]);
        //     println!("Rank {} received value: {:?}.", world.rank(), &a[..]);
        //     assert_eq!(&a[..], &[2, 4, 8, 16]);
        // });
    }
}

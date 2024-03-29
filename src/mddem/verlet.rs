use std::ops::DerefMut;

use super::{comm::Comm, domain::Domain, fix::Fixes};

pub struct Verlet {
    cycle_count: u32,
}
impl Verlet {
    pub fn new() -> Self {
        Verlet { cycle_count: 0 }
    }

    pub fn input(&mut self, commands: &Vec<String>, comm: &Comm) {
        for c in commands.iter() {
            let values = c.split_whitespace().collect::<Vec<&str>>();

            if values.len() > 0 {
                match values[0] {
                    "run" => {
                        if comm.rank == 0 {
                            println!("Verlet: {}", c);
                        }
                        self.cycle_count = values[1].parse::<u32>().unwrap();
                    }

                    _ => {}
                }
            }
        }
    }

    pub fn run(&mut self, comm: &mut Comm, domain: &mut Domain, fixes: &mut Fixes) {
        for fix in fixes.boxed_fixes.iter_mut() {
            fix.inital_integration();
        }

        for fix in fixes.boxed_fixes.iter_mut() {
            fix.pre_force();
        }

        for fix in fixes.boxed_fixes.iter_mut() {
            fix.force();
        }

        for fix in fixes.boxed_fixes.iter_mut() {
            fix.post_force();
        }

        for fix in fixes.boxed_fixes.iter_mut() {
            fix.final_integration();
        }
    }
}

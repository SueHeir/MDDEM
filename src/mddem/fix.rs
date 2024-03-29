use std::any::Any;

use super::comm::Comm;

mod gravity;

pub struct Fixes {
    pub boxed_fixes: Vec<Box<dyn Fix>>,
}

impl Fixes {
    pub fn new() -> Self {
        Fixes {
            boxed_fixes: Vec::new(),
        }
    }

    fn register_fix(&mut self, service: Box<dyn Fix>) {
        self.boxed_fixes.push(service)
    }

    // fn get_fix<S: 'static>(&self) -> Option<&S> {
    //     self.boxed_fixes
    //         .iter()
    //         .map(|s| s.downcast_ref::<S>())
    //         .find(Option::is_some)
    //         .flatten()
    // }

    // fn get_fix_mut<S: 'static>(&mut self) -> Option<&mut S> {
    //     self.boxed_fixes
    //         .iter_mut()
    //         .map(|s| s.downcast_mut::<S>())
    //         .find(Option::is_some)
    //         .flatten()
    // }

    pub fn input(&mut self, commands: &Vec<String>, comm: &Comm) {
        for c in commands.iter() {
            let values = c.split_whitespace().collect::<Vec<&str>>();

            if values.len() > 0 {
                match values[0] {
                    "fix" => match values[1] {
                        "gravity" => {
                            let gravity = gravity::Gravity::new(values);
                            self.boxed_fixes.push(Box::new(gravity));
                        }
                        _ => {
                            if comm.rank == 0 {
                                println!("Please Include a Fix Name and coresponding input values for the command.\n If you are adding a new Fix please edit the fix.rs to include this new fix")
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
    }
}

pub trait Fix {
    fn pre_inital_integration(&self);
    fn inital_integration(&self);
    fn post_inital_integration(&self);
    fn pre_exchange(&self);
    fn pre_neighbor(&self);
    fn pre_force(&self);
    fn force(&self);
    fn post_force(&self);
    fn pre_final_integration(&self);
    fn final_integration(&self);
    fn post_final_integration(&self);
}

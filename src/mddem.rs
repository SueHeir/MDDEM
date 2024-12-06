use print::print_app;
use scheduler::{ScheduleSet::*, Scheduler};
use verlet::Verlet;

mod atom;
mod comm;
mod domain;
mod neighbor;
mod force;

mod fix;
mod input;
mod print;

mod scheduler;
mod verlet;


pub struct MDDEM {
    pub scheduler: scheduler::Scheduler,
}

impl MDDEM {
    pub fn new(args: Vec<String>) -> Self {
        let scheduler = Scheduler::default();
        let mut mddem = MDDEM {
            scheduler: scheduler,
        };

        mddem.scheduler.add_resource(input::Input::new(args));

        comm::comm_app(&mut mddem.scheduler);
        neighbor::neighbor_app(&mut mddem.scheduler);
        force::force_app(&mut mddem.scheduler);
        domain::domain_app(&mut mddem.scheduler);

        atom::atom_app(&mut mddem.scheduler);

        verlet::verlet_app(&mut mddem.scheduler);
        print::print_app(&mut mddem.scheduler);

        mddem.scheduler.organize_systems();
        return mddem;
    }

    pub fn setup(&mut self) {
        self.scheduler.setup();
        // self.fixes.input(&self.input.commands, &self.comm);
    }

    pub fn run(&mut self) {
        verlet::run(&mut self.scheduler);
    }
}

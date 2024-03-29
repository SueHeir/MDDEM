mod atom;
mod comm;
mod domain;
mod fix;
mod input;
mod verlet;

pub struct MDDEM {
    input: input::Input,
    comm: comm::Comm,
    domain: domain::Domain,
    fixes: fix::Fixes,
    verlet: verlet::Verlet,
}

impl MDDEM {
    pub fn new(args: Vec<String>) -> Self {
        MDDEM {
            input: input::Input::new(args),
            comm: comm::Comm::new(),
            domain: domain::Domain::new(),
            fixes: fix::Fixes::new(),
            verlet: verlet::Verlet::new(),
        }
    }

    pub fn setup(&mut self) {
        self.comm.input(&self.input.commands);

        self.domain.input(&self.input.commands, &self.comm);

        self.fixes.input(&self.input.commands, &self.comm);

        self.verlet.input(&self.input.commands, &self.comm);
    }

    pub fn run(&mut self) {
        self.verlet
            .run(&mut self.comm, &mut self.domain, &mut self.fixes);
    }
}

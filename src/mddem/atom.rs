use nalgebra::{UnitQuaternion, Vector3};

use super::comm::Comm;

struct Atom {
    natoms: u64,
    nlocal: u32,
    nghost: u32,

    tag: Vec<u32>,

    pos: Vec<Vector3<f64>>,
    quaterion: Vec<UnitQuaternion<f64>>,
    omega: Vec<Vector3<f64>>,
    angular_momentum: Vec<Vector3<f64>>,
    torque: Vec<Vector3<f64>>,

    radius: Vec<f64>,
    rmass: Vec<f64>,
    denisty: Vec<f64>,
    mass: Vec<f64>,
}

impl Atom {
    pub fn new() -> Self {
        Atom {
            natoms: 0,
            nlocal: 0,
            nghost: 0,
            tag: Vec::new(),
            pos: Vec::new(),
            quaterion: Vec::new(),
            omega: Vec::new(),
            angular_momentum: Vec::new(),
            torque: Vec::new(),
            radius: Vec::new(),
            rmass: Vec::new(),
            denisty: Vec::new(),
            mass: Vec::new(),
        }
    }

    pub fn input(&mut self, commands: &Vec<String>, comm: &Comm) {
        for c in commands.iter() {
            let values = c.split_whitespace().collect::<Vec<&str>>();

            if values.len() > 0 {
                match values[0] {
                    "randomparticleinsert" => {
                        if comm.rank == 0 {
                            println!("Atom: {}", c);
                        }

                        if values.len() != 7 {
                            if comm.rank == 0 {
                                println!("Domain: Please fill out domain command with x_low x_high y_low y_high z_low z_high");
                            }
                            exit(1);
                        }

                        self.boundaries_low.x = values[1].parse::<f64>().unwrap();
                        self.boundaries_low.y = values[3].parse::<f64>().unwrap();
                        self.boundaries_low.z = values[5].parse::<f64>().unwrap();

                        self.boundaries_high.x = values[2].parse::<f64>().unwrap();
                        self.boundaries_high.y = values[4].parse::<f64>().unwrap();
                        self.boundaries_high.z = values[6].parse::<f64>().unwrap();
                    }

                    "perodic" => {
                        if comm.rank == 0 {
                            println!("Domain: {}", c);
                        }

                        if values.len() != 4 {
                            if comm.rank == 0 {
                                println!("Domain: Please fill out periodic command as perodic    p p p  or perodic    n n n");
                            }
                            exit(1);
                        }
                        if values[1] == "p" {
                            self.is_periodic.x = true;
                        }
                        if values[2] == "p" {
                            self.is_periodic.y = true;
                        }
                        if values[3] == "p" {
                            self.is_periodic.z = true;
                        }
                    }

                    _ => {}
                }
            }
        }
    }
}

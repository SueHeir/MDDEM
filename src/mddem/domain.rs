use std::process::exit;

use nalgebra::Vector3;

use super::comm::Comm;

pub(crate) struct Domain {
    boundaries_low: Vector3<f64>,
    boundaries_high: Vector3<f64>,
    volume: f64,
    size: Vector3<f64>,
    is_periodic: Vector3<bool>,
}

impl Domain {
    pub fn new() -> Self {
        Domain {
            boundaries_high: Vector3::new(1.0, 1.0, 1.0),
            boundaries_low: Vector3::new(0.0, 0.0, 0.0),
            size: Vector3::new(1.0, 1.0, 1.0),
            is_periodic: Vector3::new(false, false, false),
            volume: 1.0,
        }
    }

    pub fn input(&mut self, commands: &Vec<String>, comm: &Comm) {
        for c in commands.iter() {
            let values = c.split_whitespace().collect::<Vec<&str>>();

            if values.len() > 0 {
                match values[0] {
                    "domain" => {
                        if comm.rank == 0 {
                            println!("Domain: {}", c);
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

    fn volume_size_calculation(&mut self) {}
}

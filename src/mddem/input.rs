use std::{
    fs::File,
    io::{BufRead, BufReader},
};

pub struct Input {
    filename: String,
    pub commands: Vec<String>,
}

impl Input {
    pub fn new(args: Vec<String>) -> Self {
        Input {
            filename: args[1].clone(),
            commands: Self::read_input_file(&args[1]),
        }
    }

    pub fn setup(&self) {}

    fn read_input_file(filename: &String) -> Vec<String> {
        let filenamestring: String = filename.clone();
        let file = match File::open(filename) {
            Ok(file) => file,
            Err(err) => {
                println!("Error: {}", err);
                std::process::exit(1);
            }
        };

        println!("Opened File: {}", filenamestring);

        let buf = BufReader::new(file);
        buf.lines()
            .map(|l| l.expect("Could not parse line"))
            .collect()
    }
}

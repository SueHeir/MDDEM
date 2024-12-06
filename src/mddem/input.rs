use std::{
    fs::File,
    io::{BufRead, BufReader},
};

pub struct Input {
    _filename: String,
    pub commands: Vec<String>,
}

impl Input {
    pub fn new(args: Vec<String>) -> Self {
        Input {
            _filename: args[1].clone(),
            commands: Self::read_input_file(&args[1]),
        }
    }

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

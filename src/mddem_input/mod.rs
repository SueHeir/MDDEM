use std::{
    fs::File,
    io::{BufRead, BufReader},
    env,
};

use mddem_app::prelude::*;


pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        let args: Vec<String> = env::args().collect();
        app.add_resource(Input::new(args));
    }
}




pub struct Input {
    _filename: String,
    pub all_commands: Vec<String>,
    pub current_commands: Vec<Vec<String>>,

}

impl Input {
    pub fn new(args: Vec<String>) -> Self {
        let all_commands = Self::read_input_file(&args[1]);

        let mut current_commands: Vec<Vec<String>> = Vec::new();

        current_commands.push(Vec::new());
        let mut index = 0;
        for line in &all_commands {
            current_commands[index].push(line.clone());

            if line.contains("run") {
                index += 1;
                current_commands.push(Vec::new());
            }
        }


        Input {
            _filename: args[1].clone(),
            all_commands,
            current_commands,
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
        let args = buf.lines()
            .map(|l| l.expect("Could not parse line"))
            .collect();

        println!("Finished reading File");
        return args;
    }
}

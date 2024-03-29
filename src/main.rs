/*
Molecular Dynamics and Discrete Element Method
By Elizabeth Suehr
*/
mod mddem;
use std::env;

use mddem::MDDEM;

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("{:?}", args);

    let mut mddem = MDDEM::new(args);

    mddem.setup();

    mddem.run();
}

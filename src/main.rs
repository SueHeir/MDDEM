/*
Molecular Dynamics and Discrete Element Method
By Elizabeth Suehr
*/
mod mddem;

use mddem::MDDEM;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("{:?}", args);
    let mut mddem = MDDEM::new(args);

    mddem.setup();

    mddem.run();
}

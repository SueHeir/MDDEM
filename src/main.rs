/*
Molecular Dynamics and Discrete Element Method
By Elizabeth Suehr
*/

pub use mddem_app::prelude::*;

mod mddem_input;
mod mddem_domain;
mod mddem_communication;
mod mddem_neighbor;
mod mddem_atom;
mod mddem_force;
mod mddem_verlet;
mod mddem_print;



use mddem_input::InputPlugin;
use mddem_domain::DomainPlugin;
use mddem_communication::CommincationPlugin;
use mddem_neighbor::NeighborPlugin;
use mddem_atom::AtomPlugin;
use mddem_force::ForcePlugin;
use mddem_verlet::VeletPlugin;
use mddem_print::PrintPlugin;



fn main() {
    App::new()
        .add_plugins(InputPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(CommincationPlugin)
        .add_plugins(NeighborPlugin)
        .add_plugins(AtomPlugin)
        .add_plugins(ForcePlugin)
        .add_plugins(VeletPlugin)
        .add_plugins(PrintPlugin)
        .start();

}

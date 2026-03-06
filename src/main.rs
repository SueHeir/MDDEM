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
mod dem_atom;
mod dem_granular;
mod mddem_verlet;
mod mddem_print;

use mddem_input::InputPlugin;
use mddem_domain::DomainPlugin;
use mddem_communication::CommincationPlugin;
use mddem_neighbor::NeighborPlugin;
use dem_granular::GranularDefaultPlugins;
use mddem_verlet::VerletPlugin;
use mddem_print::PrintPlugin;

fn main() {
    let mut app = App::new();
    app.add_plugins(InputPlugin)
        .add_plugins(CommincationPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(NeighborPlugin { brute_force: false })
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(VerletPlugin)
        .add_plugins(PrintPlugin);

    if std::env::args().any(|a| a == "--schedule") {
        app.enable_schedule_print();
    }

    app.start();
}

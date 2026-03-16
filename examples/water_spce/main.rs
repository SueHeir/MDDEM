//! SPC/E water model proof-of-concept: O-O Lennard-Jones only.
//!
//! This example simulates LJ particles at liquid-state conditions using the
//! SPC/E oxygen-oxygen LJ parameters in reduced units. All atoms represent
//! oxygen sites; charges and bonds are omitted for simplicity.
//!
//! The resulting g(r) shows liquid-like structure (first peak ~1.0 sigma,
//! coordination shell). For a full SPC/E simulation, electrostatic
//! interactions and rigid O-H bonds would be needed.
//!
//! Run with:
//!   cargo run --example water_spce --no-default-features -- -c examples/water_spce/config.toml

use mddem::prelude::*;
use md_water::WaterPlugin;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(VelocityVerletPlugin::new())
        .add_plugins(LatticePlugin)
        .add_plugins(LJForcePlugin)
        .add_plugins(LangevinPlugin)
        .add_plugins(MeasurePlugin)
        .add_plugins(WaterPlugin);
    app.start();
}

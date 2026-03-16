//! Example template: minimal MDDEM simulation structure.
//!
//! This template demonstrates the bare minimum code needed for a DEM
//! granular simulation. Copy this directory and modify it for your use case.
//!
//! # Running
//! ```bash
//! cargo run --example example_template --no-default-features -- examples/example_template/config.toml
//! ```
//!
//! # What each plugin provides
//! - `CorePlugins` — CLI parsing, config loading, domain setup, neighbor lists, output
//! - `GranularDefaultPlugins` — DEM atoms, Hertz-Mindlin contact, rotational dynamics
//! - `GravityPlugin` — constant gravitational acceleration (from `[gravity]` config)
//! - `WallPlugin` — planar/cylinder/sphere wall boundaries (from `[[wall]]` config)
//!
//! # Customizing
//! - Add `FixesPlugin` for freeze, addforce, setforce, move_linear, viscous fixes
//! - Add `DemBondPlugin` for bonded particle models
//! - Add `ThermalPlugin` for heat conduction between particles
//! - Add `StatesPlugin` + `StageAdvancePlugin` for multi-stage simulations
//! - See the `hopper` or `conveyor_belt` examples for advanced usage

use mddem::prelude::*;

fn main() {
    let mut app = App::new();

    // ── Infrastructure + DEM physics ──────────────────────────────────────
    // CorePlugins:            input, communication, domain, neighbor, run, print
    // GranularDefaultPlugins: DEM atoms, velocity verlet, Hertz-Mindlin contact,
    //                         rotational dynamics, granular temperature output
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins);

    // ── Optional plugins (uncomment as needed) ───────────────────────────
    // app.add_plugins(GravityPlugin);      // needs [gravity] in config
    // app.add_plugins(WallPlugin);         // needs [[wall]] in config
    // app.add_plugins(FixesPlugin);        // needs [[addforce]], [[freeze]], etc.

    app.start();
}

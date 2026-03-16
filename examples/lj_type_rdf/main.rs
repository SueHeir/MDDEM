//! LJ liquid with type-filtered RDF measurement.
//!
//! Demonstrates the `TypeRdfPlugin` for measuring g(r) between specific atom
//! type pairs. This example uses SPC/E oxygen-oxygen LJ parameters in reduced
//! units as a proof-of-concept, but the type-filtered RDF works for any
//! multi-type simulation (binary mixtures, alloys, etc.).
//!
//! Run with:
//!   cargo run --example lj_type_rdf --no-default-features -- -c examples/lj_type_rdf/config.toml

use mddem::prelude::*;
use md_type_rdf::TypeRdfPlugin;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(VelocityVerletPlugin::new())
        .add_plugins(LatticePlugin)
        .add_plugins(LJForcePlugin)
        .add_plugins(LangevinPlugin)
        .add_plugins(MeasurePlugin)
        .add_plugins(TypeRdfPlugin);
    app.start();
}

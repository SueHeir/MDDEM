//! Binary Lennard-Jones mixture (Kob-Andersen model).
//!
//! Classic 80:20 binary mixture that forms a glass at low temperature.
//! Uses per-type-pair LJ parameters and measures partial RDFs g_AA(r),
//! g_AB(r), g_BB(r) along with per-type mean-squared displacement.
//!
//! Kob-Andersen parameters (LJ reduced units):
//! - A-A: eps=1.0, sigma=1.0
//! - A-B: eps=1.5, sigma=0.8
//! - B-B: eps=0.5, sigma=0.88

use mddem::prelude::*;

fn main() {
    let mut app = App::new();

    app.add_plugins(CorePlugins)
        .add_plugins(VelocityVerletPlugin::new())
        .add_plugins(LatticePlugin)
        .add_plugins(LJForcePlugin)
        .add_plugins(LangevinPlugin)
        .add_plugins(MeasurePlugin)
        .add_plugins(TypeRdfPlugin)
        .add_plugins(md_msd::TypeMsdPlugin);

    app.start();
}

//! Conveyor belt DEM example.
//!
//! Demonstrates:
//! - A flat wall with prescribed velocity acting as a conveyor belt surface
//! - Rate-based particle insertion dropping particles onto the belt
//! - Measurement plane throughput tracking (particles/s, kg/s)
//!
//! The belt is a horizontal plane wall at z=0 with velocity in the +x direction.
//! Particles are inserted above the belt and fall under gravity onto the moving surface.
//! A measurement plane at the belt exit counts particles passing through.

use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins);
    app.add_plugins(GranularDefaultPlugins);
    app.add_plugins(GravityPlugin);
    app.add_plugins(WallPlugin);
    app.add_plugins(FixesPlugin);
    app.add_plugins(MeasurePlanePlugin);
    app.start();
}

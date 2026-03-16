use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(VelocityVerletPlugin::new())
        .add_plugins(LJForcePlugin)
        .add_plugins(LangevinPlugin)
        .add_plugins(PolymerPlugin)
        .add_plugins(MdBondPlugin);
    app.start();
}

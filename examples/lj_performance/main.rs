use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(LatticePlugin)
        .add_plugins(LJForcePlugin)
        .add_plugins(NoseHooverPlugin);
    app.start();
}

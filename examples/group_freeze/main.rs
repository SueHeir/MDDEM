use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(LJDefaultPlugins)
        .add_plugins(FixesPlugin);
    app.start();
}

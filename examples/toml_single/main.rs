//! Quick test: 500 particles, 10,000 steps — config built programmatically.
//!
//! Demonstrates the Rust API tier: no TOML file needed, everything is
//! constructed in code with full type safety.
//!
//! ```bash
//! cargo run --example toml_single --no-default-features
//! ```

use mddem::prelude::*;

fn main() {
    let mut table = toml::Table::new();

    let mut domain = toml::Table::new();
    domain.insert("x_high".into(), toml::Value::Float(0.025));
    domain.insert("y_high".into(), toml::Value::Float(0.025));
    domain.insert("z_high".into(), toml::Value::Float(0.025));
    domain.insert("periodic_x".into(), true.into());
    domain.insert("periodic_y".into(), true.into());
    domain.insert("periodic_z".into(), true.into());
    table.insert("domain".into(), domain.into());

    let mut neighbor = toml::Table::new();
    neighbor.insert("skin_fraction".into(), toml::Value::Float(1.1));
    neighbor.insert("bin_size".into(), toml::Value::Float(0.005));
    table.insert("neighbor".into(), neighbor.into());

    let mut mat = toml::Table::new();
    mat.insert("name".into(), "glass".into());
    mat.insert("youngs_mod".into(), toml::Value::Float(8.7e9));
    mat.insert("poisson_ratio".into(), toml::Value::Float(0.3));
    mat.insert("restitution".into(), toml::Value::Float(0.95));
    mat.insert("friction".into(), toml::Value::Float(0.4));
    let mut dem = toml::Table::new();
    dem.insert(
        "materials".into(),
        toml::Value::Array(vec![toml::Value::Table(mat)]),
    );
    table.insert("dem".into(), dem.into());

    let mut insert = toml::Table::new();
    insert.insert("material".into(), "glass".into());
    insert.insert("count".into(), 500.into());
    insert.insert("radius".into(), toml::Value::Float(0.001));
    insert.insert("density".into(), toml::Value::Float(2500.0));
    insert.insert("velocity".into(), toml::Value::Float(0.5));
    let mut particles = toml::Table::new();
    particles.insert(
        "insert".into(),
        toml::Value::Array(vec![toml::Value::Table(insert)]),
    );
    table.insert("particles".into(), particles.into());

    let mut run = toml::Table::new();
    run.insert("steps".into(), 10000.into());
    run.insert("thermo".into(), 100.into());
    table.insert("run".into(), run.into());

    let mut app = App::new();
    app.add_resource(Input {
        filename: String::new(),
        output_dir: None,
    });
    app.add_resource(Config { table });
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins);
    app.start();
}

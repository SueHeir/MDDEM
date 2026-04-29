//! MDDEM-specific wrapper around [`grass_io::InputPlugin`] — adds the
//! ASCII banner and the `--schedule` CLI flag handling on top of the
//! generic config-loading flow.
//!
//! Re-exports [`Config`], [`Input`], and [`load_toml`] from `grass_io`
//! so that downstream MDDEM code continues to write
//! `mddem_core::Config` etc. unchanged after the upstreaming of the
//! input/run machinery.

use std::env;

use grass_app::prelude::*;

pub use grass_io::{load_toml, Config, Input};

/// Plugin that adds the MDDEM ASCII banner, then delegates to
/// [`grass_io::InputPlugin`] for CLI parsing + TOML loading. Also
/// honors the MDDEM-specific `--schedule` flag (dumps the compiled
/// schedule to a DOT file).
///
/// If a [`Config`] resource is already present (programmatic use),
/// CLI parsing is skipped — this plugin no-ops, matching
/// `grass_io::InputPlugin`'s seed-skip behavior.
pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        if app.get_resource_ref::<Config>().is_some() {
            return;
        }

        let args: Vec<String> = env::args().collect();
        let schedule = args.iter().any(|a| a == "--schedule");

        print_banner();

        app.add_plugins(grass_io::InputPlugin);

        if schedule {
            app.enable_schedule_print();
        }
    }
}

/// Print the MDDEM ASCII banner (rank-0 only, detected via MPI env vars).
pub fn print_banner() {
    let is_rank0 = match env::var("OMPI_COMM_WORLD_RANK")
        .or_else(|_| env::var("PMI_RANK"))
        .or_else(|_| env::var("PMIX_RANK"))
    {
        Ok(r) => r == "0",
        Err(_) => true,
    };
    if is_rank0 {
        println!();
        println!("  o    o  o-o    o-o   o--o  o    o ");
        println!("  |\\  /| |   \\  |   \\  |     |\\  /| ");
        println!("  | \\/ | |    | |    | |--   | \\/ | ");
        println!("  |    | |   /  |   /  |     |    | ");
        println!("  o    o o-o    o-o    o--o  o    o ");
        println!("  Molecular Dynamics / Discrete Element Method");
        println!();
    }
}

//! CLI argument parsing, TOML config loading, and the startup banner.
//!
//! The [`InputPlugin`] handles:
//! - Parsing `args[1]` as the input TOML path
//! - `--schedule` flag to print the system execution order
//! - `--generate-config` flag to emit a complete example config
//! - Loading and storing the parsed [`Config`] resource

use std::env;

use sim_app::prelude::*;

mod toml_input;
pub use toml_input::Config;

/// CLI arguments: input filename and output directory path.
pub struct Input {
    pub filename: String,
    pub output_dir: Option<String>,
}

/// Plugin that handles CLI parsing, banner printing, and config loading.
///
/// If a `Config` resource is already present (programmatic use), CLI parsing is skipped.
/// Otherwise, parses `args[1]` for the input file and `--schedule` to print the schedule.
pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        // If Config already exists, the user set it up programmatically — skip CLI.
        if app.get_resource_ref::<Config>().is_some() {
            return;
        }

        let args: Vec<String> = env::args().collect();

        if args.iter().any(|a| a == "--generate-config") {
            print_banner();
            app.add_resource(Config { table: toml::Table::new() });
            app.add_resource(Input { filename: String::new(), output_dir: None });
            app.add_resource(GenerateConfigFlag);
            return;
        }

        let input_file = args.get(1).cloned().unwrap_or_else(|| {
            eprintln!("Usage: mddem <input.toml> [--schedule] [--generate-config]");
            std::process::exit(1);
        });
        let schedule = args.iter().any(|a| a == "--schedule");

        print_banner();
        let table = load_toml(&input_file);

        // Output directory: prefer [output] dir from config, else use config file's parent
        let output_dir = table
            .get("output")
            .and_then(|v| v.as_table())
            .and_then(|t| t.get("dir"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                std::path::Path::new(&input_file)
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .map(|p| p.to_string_lossy().into_owned())
            });
        app.add_resource(Input {
            filename: input_file,
            output_dir,
        });
        app.add_resource(Config { table });
        if schedule {
            app.enable_schedule_print();
        }
    }
}

/// Print the MDDEM ASCII banner (rank-0 only).
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

/// Read and parse a TOML input file, returning the parsed table.
pub fn load_toml(path: &str) -> toml::Table {
    if !path.ends_with(".toml") {
        eprintln!("Error: input file must be a .toml file, got '{}'", path);
        std::process::exit(1);
    }
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading '{}': {}", path, e);
        std::process::exit(1);
    });
    toml::from_str(&content).unwrap_or_else(|e| {
        eprintln!("Error parsing TOML '{}': {}", path, e);
        std::process::exit(1);
    })
}

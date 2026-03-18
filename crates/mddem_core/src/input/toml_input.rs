//! TOML configuration parsing utilities.
//!
//! [`Config`] wraps a parsed `toml::Table` and provides typed section extraction
//! with user-friendly error messages pointing to the offending field.

use std::any::TypeId;

use sim_app::prelude::App;
use serde::Deserialize;

/// Wraps a parsed TOML table. Use [`Config::load::<T>(app, key)`] to extract and register a section.
pub struct Config {
    pub table: toml::Table,
}

impl Config {
    /// Deserialize a `[key]` section from the TOML table, returning `T::default()` if absent.
    ///
    /// Prints an actionable error and exits if deserialization fails.
    pub fn section<T: for<'de> Deserialize<'de> + Default>(&self, key: &str) -> T {
        match self.table.get(key) {
            None => T::default(),
            Some(v) => match v.clone().try_into::<T>() {
                Ok(val) => val,
                Err(e) => {
                    eprintln!();
                    eprintln!("ERROR: Failed to parse [{}] section in config file.", key);
                    eprintln!("  {}", e);
                    eprintln!();
                    eprintln!("  Hint: Check that all field names are spelled correctly and values have the right type.");
                    eprintln!("  Run with --generate-config to see a complete example configuration.");
                    std::process::exit(1);
                }
            },
        }
    }

    /// Extract a config section and register it as an ECS resource on `app`.
    ///
    /// Looks up `[key]` in the TOML table, deserializes it into `T`, clones it
    /// into the app as a resource, and returns the value.
    pub fn load<T: for<'de> Deserialize<'de> + Default + Clone + 'static>(
        app: &mut App,
        key: &str,
    ) -> T {
        let config = if let Some(raw_cell) = app.get_mut_resource(TypeId::of::<Config>()) {
            let raw = raw_cell.borrow();
            let raw_config = raw
                .downcast_ref::<Config>()
                .expect("Config resource has wrong type — this is a bug in MDDEM");
            let c: T = raw_config.section(key);
            drop(raw);
            c
        } else {
            T::default()
        };
        app.add_resource(config.clone());
        config
    }

    /// Parse the `run` key as either `[run]` (single table) or `[[run]]` (array of tables).
    pub fn parse_run_config(&self) -> crate::RunConfig {
        use crate::{RunConfig, StageConfig};
        match self.table.get("run") {
            Some(toml::Value::Array(arr)) => {
                let stages: Vec<StageConfig> = arr
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| match v.clone().try_into::<StageConfig>() {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!();
                            eprintln!("ERROR: Failed to parse [[run]] stage {} in config file.", idx);
                            eprintln!("  {}", e);
                            eprintln!();
                            eprintln!("  Hint: Check that all field names are spelled correctly and values have the right type.");
                            eprintln!("  Run with --generate-config to see a complete example configuration.");
                            std::process::exit(1);
                        }
                    })
                    .collect();
                RunConfig { stages }
            }
            Some(toml::Value::Table(_)) => {
                let stage: StageConfig = self.section("run");
                RunConfig {
                    stages: vec![stage],
                }
            }
            _ => RunConfig::default(),
        }
    }

    /// Parse a `[[key]]` TOML array into a `Vec<T>`.
    pub fn parse_array<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Vec<T> {
        match self.table.get(key) {
            Some(toml::Value::Array(arr)) => arr
                .iter()
                .enumerate()
                .map(|(idx, v)| match v.clone().try_into::<T>() {
                    Ok(val) => val,
                    Err(e) => {
                        eprintln!();
                        eprintln!("ERROR: Failed to parse [[{}]] entry {} in config file.", key, idx);
                        eprintln!("  {}", e);
                        eprintln!();
                        eprintln!("  Hint: Check that all field names are spelled correctly and values have the right type.");
                        eprintln!("  Run with --generate-config to see a complete example configuration.");
                        std::process::exit(1);
                    }
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Load RunConfig from App's Config resource, handling [run] vs [[run]] syntax.
    pub fn load_run_config(app: &mut App) -> crate::RunConfig {
        if let Some(raw_cell) = app.get_mut_resource(TypeId::of::<Config>()) {
            let raw = raw_cell.borrow();
            let config = raw
                .downcast_ref::<Config>()
                .expect("Config resource has wrong type — this is a bug in MDDEM");
            let rc = config.parse_run_config();
            drop(raw);
            rc
        } else {
            crate::RunConfig::default()
        }
    }

    /// Load a config section from the stage-aware merged config (`StageOverrides`).
    ///
    /// This reads from the merged table (global defaults + current stage overrides),
    /// so per-stage overrides like `thermostat.temperature = 1.2` inside `[[run]]` apply.
    pub fn load_stage_aware<T: for<'de> Deserialize<'de> + Default>(
        stage_overrides: &crate::StageOverrides,
        key: &str,
    ) -> T {
        stage_overrides.section(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_run_table() {
        let toml_str = r#"
[run]
steps = 5000
thermo = 200
"#;
        let table: toml::Table = toml_str.parse().unwrap();
        let config = Config { table };
        let run = config.parse_run_config();
        assert_eq!(run.num_stages(), 1);
        assert_eq!(run.current_stage(0).steps, 5000);
        assert_eq!(run.current_stage(0).thermo, 200);
        assert!(run.current_stage(0).name.is_none());
    }

    #[test]
    fn parses_multi_run_array() {
        let toml_str = r#"
[[run]]
name = "settling"
steps = 1000
thermo = 100

[[run]]
name = "production"
steps = 5000
thermo = 500
dump_interval = 100
"#;
        let table: toml::Table = toml_str.parse().unwrap();
        let config = Config { table };
        let run = config.parse_run_config();
        assert_eq!(run.num_stages(), 2);
        assert_eq!(run.current_stage(0).name.as_deref(), Some("settling"));
        assert_eq!(run.current_stage(0).steps, 1000);
        assert_eq!(run.current_stage(0).thermo, 100);
        assert_eq!(run.current_stage(1).name.as_deref(), Some("production"));
        assert_eq!(run.current_stage(1).steps, 5000);
        assert_eq!(run.current_stage(1).thermo, 500);
        assert_eq!(run.current_stage(1).dump_interval, Some(100));
        assert!(run.current_stage(0).dump_interval.is_none());
    }

    #[test]
    fn default_fallback_for_missing_section() {
        let config = Config {
            table: toml::Table::new(),
        };
        let run = config.parse_run_config();
        assert_eq!(run.num_stages(), 1);
        assert_eq!(run.current_stage(0).steps, 1000);
        assert_eq!(run.current_stage(0).thermo, 100);
    }

    #[test]
    fn stage_config_captures_overrides() {
        let toml_str = r#"
[[run]]
name = "settle"
steps = 1000
thermo = 100

[[run]]
name = "compress"
steps = 5000
thermo = 500
thermostat.temperature = 1.2
"#;
        let table: toml::Table = toml_str.parse().unwrap();
        let config = Config { table };
        let run = config.parse_run_config();
        assert_eq!(run.num_stages(), 2);
        // First stage has no overrides
        assert!(run.current_stage(0).overrides.is_empty());
        // Second stage has thermostat override
        let overrides = &run.current_stage(1).overrides;
        assert!(overrides.contains_key("thermostat"));
        let thermostat = overrides["thermostat"].as_table().unwrap();
        assert_eq!(thermostat["temperature"].as_float(), Some(1.2));
    }

    #[test]
    fn deep_merge_works() {
        use crate::run::deep_merge;
        let mut base: toml::Table = r#"
[thermostat]
temperature = 0.85
damping = 100.0
"#
        .parse()
        .unwrap();
        let overrides: toml::Table = r#"
[thermostat]
temperature = 1.2
"#
        .parse()
        .unwrap();
        deep_merge(&mut base, &overrides);
        let thermostat = base["thermostat"].as_table().unwrap();
        assert_eq!(thermostat["temperature"].as_float(), Some(1.2));
        assert_eq!(thermostat["damping"].as_float(), Some(100.0));
    }

    #[test]
    fn stage_overrides_section() {
        use crate::StageOverrides;
        let table: toml::Table = r#"
[thermostat]
temperature = 1.2
damping = 100.0
"#
        .parse()
        .unwrap();
        let so = StageOverrides { table };

        #[derive(Deserialize, Default)]
        struct ThermoConfig {
            #[serde(default)]
            temperature: f64,
            #[serde(default)]
            damping: f64,
        }
        let tc: ThermoConfig = so.section("thermostat");
        assert!((tc.temperature - 1.2).abs() < 1e-10);
        assert!((tc.damping - 100.0).abs() < 1e-10);
    }
}

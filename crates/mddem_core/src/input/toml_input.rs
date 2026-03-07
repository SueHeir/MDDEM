use std::any::TypeId;

use serde::Deserialize;
use mddem_app::prelude::App;

pub struct Config {
    pub table: toml::Table,
}

impl Config {
    pub fn section<T: for<'de> Deserialize<'de> + Default>(&self, key: &str) -> T {
        self.table
            .get(key)
            .and_then(|v| v.clone().try_into().ok())
            .unwrap_or_default()
    }

    pub fn load<T: for<'de> Deserialize<'de> + Default + Clone + 'static>(app: &mut App, key: &str) -> T {
        let config = if let Some(raw_cell) = app.get_mut_resource(TypeId::of::<Config>()) {
            let raw = raw_cell.borrow();
            let raw_config = raw.downcast_ref::<Config>().unwrap();
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
                let stages: Vec<StageConfig> = arr.iter()
                    .map(|v| v.clone().try_into().unwrap_or_default())
                    .collect();
                RunConfig { stages }
            }
            Some(toml::Value::Table(_)) => {
                let stage: StageConfig = self.section("run");
                RunConfig { stages: vec![stage] }
            }
            _ => RunConfig::default(),
        }
    }

    /// Load RunConfig from App's Config resource, handling [run] vs [[run]] syntax.
    pub fn load_run_config(app: &mut App) -> crate::RunConfig {
        if let Some(raw_cell) = app.get_mut_resource(TypeId::of::<Config>()) {
            let raw = raw_cell.borrow();
            let config = raw.downcast_ref::<Config>().unwrap();
            let rc = config.parse_run_config();
            drop(raw);
            rc
        } else {
            crate::RunConfig::default()
        }
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
        let config = Config { table: toml::Table::new() };
        let run = config.parse_run_config();
        assert_eq!(run.num_stages(), 1);
        assert_eq!(run.current_stage(0).steps, 1000);
        assert_eq!(run.current_stage(0).thermo, 100);
    }
}

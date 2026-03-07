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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RunConfig;

    #[test]
    fn parses_run_config_correctly() {
        let toml_str = r#"
[run]
steps = 5000
thermo = 200
"#;
        let table: toml::Table = toml_str.parse().unwrap();
        let config = Config { table };
        let run: RunConfig = config.section("run");
        assert_eq!(run.steps, 5000);
        assert_eq!(run.thermo, 200);
    }

    #[test]
    fn default_fallback_for_missing_section() {
        let config = Config { table: toml::Table::new() };
        let run: RunConfig = config.section("run");
        assert_eq!(run.steps, 1000);
        assert_eq!(run.thermo, 100);
    }
}

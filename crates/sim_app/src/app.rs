//! The central [`App`] container and its supporting types.
//!
//! [`App`] is the entry point for every simulation. It owns the main
//! [`SubApp`], coordinates plugin registration, and drives the simulation
//! lifecycle (setup → run → cleanup).
//!
//! # Typical usage
//!
//! ```rust,ignore
//! use sim_app::prelude::*;
//!
//! App::new()
//!     .add_plugins(MyPlugins)
//!     .start();
//! ```

use std::{
    any::{Any, TypeId},
    cell::RefCell,
};

use sim_scheduler::{IntoScheduledSystem, IntoSystem, ScheduleSet};

use crate::{Plugin, Plugins, SubApp, SubApps};

/// Collected TOML snippets from all plugins that implement [`Plugin::default_config`].
///
/// This resource is automatically populated during plugin registration. When the
/// [`GenerateConfigFlag`] resource is present, [`App::start`] prints these
/// snippets to stdout and exits.
pub struct ConfigSnippets {
    /// The accumulated TOML snippet strings, one per plugin.
    pub snippets: Vec<String>,
}

/// Marker resource: when present, [`App::start`] prints config snippets and exits
/// instead of running the simulation.
///
/// Add this resource (e.g. via a `--generate-config` CLI flag) to have the app
/// emit a complete example configuration file assembled from all registered plugins.
pub struct GenerateConfigFlag;

/// Central application container. Holds resources, systems, and plugins.
///
/// `App` provides a builder-style API for assembling a simulation from plugins.
/// Most methods return `&mut Self` so calls can be chained:
///
/// ```rust,ignore
/// App::new()
///     .add_plugins(PhysicsPlugins)
///     .add_resource(MyConfig { dt: 0.001 })
///     .add_update_system(my_system, ScheduleSet::Update)
///     .start();
/// ```
pub struct App {
    pub(crate) sub_apps: SubApps,
    cleanup_fns: Vec<Box<dyn FnOnce()>>,
}

impl Default for App {
    fn default() -> Self {
        App::new()
    }
}

impl App {
    /// Creates a new, empty [`App`] with default structure.
    ///
    /// This is the preferred constructor for most use cases. After creation,
    /// add plugins via [`add_plugins`](Self::add_plugins) and start the
    /// simulation with [`start`](Self::start).
    pub fn new() -> App {
        Self {
            sub_apps: SubApps {
                main: SubApp::new(),
            },
            cleanup_fns: Vec::new(),
        }
    }

    /// Registers one or more plugins with this app.
    ///
    /// Accepts any type implementing [`Plugin`], [`PluginGroup`](crate::PluginGroup),
    /// or a tuple of plugins.
    ///
    /// # Panics
    ///
    /// Panics if a unique plugin is added twice or if plugin dependencies are
    /// not satisfied. The panic message includes guidance on how to fix the
    /// registration order.
    pub fn add_plugins<M>(&mut self, plugins: impl Plugins<M>) -> &mut Self {
        plugins.add_to_app(self);
        self
    }

    /// Internal: adds a boxed plugin, checking uniqueness and dependencies.
    pub(crate) fn add_boxed_plugin(
        &mut self,
        plugin: Box<dyn Plugin>,
    ) -> Result<&mut Self, AppError> {
        if plugin.is_unique() && self.main_mut().plugin_names.contains(plugin.name()) {
            return Err(AppError::DuplicatePlugin {
                plugin_name: plugin.name().to_string(),
            });
        }

        self.validate_dependencies(&*plugin)?;

        // Record the plugin's TypeId for TypeId-based dependency checks.
        let plugin_type_id = (*plugin).type_id();
        self.main_mut().plugin_type_ids.insert(plugin_type_id);

        // Record the plugin name *before* build so that nested add_plugins calls
        // within build() can see this plugin as registered (prevents false-positive
        // dependency errors when a plugin group adds a dependency and its dependent
        // in sequence).
        let plugin_name = plugin.name().to_string();
        self.main_mut()
            .plugin_names
            .insert(plugin_name.clone());

        plugin.build(self);

        self.collect_config_snippet(&*plugin);

        // Collect capability contracts after build.
        for cap in plugin.provides() {
            self.main_mut()
                .provided_capabilities
                .insert(cap.to_string());
        }
        for cap in plugin.requires() {
            self.main_mut()
                .required_capabilities
                .push((cap.to_string(), plugin_name.clone()));
        }

        Ok(self)
    }

    /// Checks that all plugins listed in `plugin.dependencies()` (by [`TypeId`])
    /// have already been registered. Returns `Err(AppError::MissingDependencies)`
    /// if any are missing.
    fn validate_dependencies(&self, plugin: &dyn Plugin) -> Result<(), AppError> {
        let deps = plugin.dependencies();
        if deps.is_empty() {
            return Ok(());
        }

        let missing: Vec<TypeId> = deps
            .into_iter()
            .filter(|dep| !self.main().plugin_type_ids.contains(dep))
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(AppError::MissingDependencies {
                plugin_name: plugin.name().to_string(),
                missing,
            })
        }
    }

    /// Validates that every required capability tag has at least one provider.
    ///
    /// Called in [`start`](Self::start) after all plugins are registered.
    ///
    /// # Panics
    ///
    /// Panics with a clear error listing all unsatisfied capabilities.
    fn validate_capability_contracts(&self) {
        let provided = &self.main().provided_capabilities;
        let required = &self.main().required_capabilities;

        let missing: Vec<_> = required
            .iter()
            .filter(|(cap, _)| !provided.contains(cap))
            .collect();

        if !missing.is_empty() {
            eprintln!();
            eprintln!("ERROR: Missing capability contracts:");
            for (cap, plugin_name) in &missing {
                eprintln!("  - capability `{}` required by `{}`", cap, plugin_name);
            }
            eprintln!();
            eprintln!("  Hint: Add a plugin that provides the missing capabilities.");
            panic!(
                "Missing capabilities: {:?}",
                missing
                    .iter()
                    .map(|(cap, _)| cap.as_str())
                    .collect::<Vec<_>>()
            );
        }
    }

    /// If the plugin provides a [`Plugin::default_config`] snippet, appends it
    /// to the [`ConfigSnippets`] resource (creating the resource if needed).
    fn collect_config_snippet(&mut self, plugin: &dyn Plugin) {
        let Some(snippet) = plugin.default_config() else {
            return;
        };
        let snippet = snippet.to_string();

        if let Some(cell) = self.get_mut_resource(TypeId::of::<ConfigSnippets>()) {
            let mut borrow = cell.borrow_mut();
            let snippets = borrow
                .downcast_mut::<ConfigSnippets>()
                .expect("ConfigSnippets resource has wrong type — this is a framework bug");
            snippets.snippets.push(snippet);
        } else {
            self.add_resource(ConfigSnippets {
                snippets: vec![snippet],
            });
        }
    }

    /// Returns a reference to the main [`SubApp`].
    pub fn main(&self) -> &SubApp {
        &self.sub_apps.main
    }

    /// Returns a mutable reference to the main [`SubApp`].
    pub fn main_mut(&mut self) -> &mut SubApp {
        &mut self.sub_apps.main
    }

    /// Organizes registered systems into their schedule-set order.
    ///
    /// Called automatically by [`start`](Self::start); you only need this if
    /// you are manually driving the setup/run cycle.
    pub fn organize_systems(&mut self) {
        self.sub_apps.main.organize_systems();
    }

    /// Runs all setup systems in their schedule-setup-set order.
    ///
    /// Called automatically by [`start`](Self::start).
    pub fn setup(&mut self) -> &mut Self {
        self.sub_apps.main.setup();
        self
    }

    /// Runs the main simulation loop (all update systems each timestep).
    ///
    /// Called automatically by [`start`](Self::start).
    pub fn run(&mut self) -> &mut Self {
        self.sub_apps.main.run();
        self
    }

    /// Registers a system to run during the setup phase at the given schedule phase.
    pub fn add_setup_system<M>(
        &mut self,
        system: impl IntoScheduledSystem<M>,
        schedule_set: impl ScheduleSet,
    ) -> &mut Self {
        self.sub_apps.main.add_setup_system(system, schedule_set);
        self
    }

    /// Registers a system to run every timestep at the given schedule phase.
    pub fn add_update_system<M>(
        &mut self,
        system: impl IntoScheduledSystem<M>,
        schedule_set: impl ScheduleSet,
    ) -> &mut Self {
        self.sub_apps.main.add_update_system(system, schedule_set);
        self
    }

    /// Assigns a namespace to all systems registered under the given phase enum type.
    ///
    /// Systems sort by `(namespace, index)`, so this controls cross-solver ordering.
    pub fn set_schedule_namespace<P: ScheduleSet + 'static>(&mut self, namespace: u32) -> &mut Self {
        self.sub_apps.main.set_schedule_namespace::<P>(namespace);
        self
    }

    /// Inserts a resource into the app's resource store.
    ///
    /// If a resource of the same type already exists, it is replaced.
    pub fn add_resource<R: 'static>(&mut self, res: R) -> &mut Self {
        self.sub_apps.main.add_resource(res);
        self
    }

    /// Returns a mutable reference to the raw resource cell for the given [`TypeId`],
    /// or `None` if no resource of that type exists.
    pub fn get_mut_resource(&mut self, res: TypeId) -> Option<&RefCell<Box<dyn Any>>> {
        self.sub_apps.main.get_mut_resource(res)
    }

    /// Returns a borrowed reference to a resource of type `R`, or `None` if it
    /// has not been added.
    pub fn get_resource_ref<R: 'static>(&self) -> Option<std::cell::Ref<'_, R>> {
        self.sub_apps.main.get_resource_ref::<R>()
    }

    /// Registers a cleanup function that will run after the simulation finishes
    /// (or after config generation). Cleanup functions run in registration order.
    pub fn add_cleanup(&mut self, f: fn()) -> &mut Self {
        self.cleanup_fns.push(Box::new(f));
        self
    }

    /// Starts the simulation lifecycle.
    ///
    /// If the [`GenerateConfigFlag`] resource is present, prints all collected
    /// config snippets to stdout and exits. Otherwise, runs
    /// [`organize_systems`](Self::organize_systems) → setup → run → cleanup.
    pub fn start(&mut self) {
        if self.get_resource_ref::<GenerateConfigFlag>().is_some() {
            self.print_generated_config();
            self.run_cleanup();
            return;
        }
        self.validate_capability_contracts();
        self.sub_apps.main.start();
        self.run_cleanup();
    }

    /// Prints accumulated config snippets from all registered plugins.
    fn print_generated_config(&self) {
        let Some(snippets) = self.get_resource_ref::<ConfigSnippets>() else {
            return;
        };
        println!("# Generated configuration");
        println!("# Default values for all registered plugins\n");
        for snippet in &snippets.snippets {
            println!("{}", snippet.trim());
            println!();
        }
    }

    /// Runs all registered cleanup functions, draining the list.
    fn run_cleanup(&mut self) {
        for f in self.cleanup_fns.drain(..) {
            f();
        }
    }

    /// Removes an update system by its concrete type.
    pub fn remove_update_system<I, S: sim_scheduler::System + 'static>(
        &mut self,
        system: impl IntoSystem<I, System = S>,
    ) -> &mut Self {
        self.sub_apps.main.remove_update_system(system);
        self
    }

    /// Removes an update system identified by its string label.
    pub fn remove_update_system_by_label(&mut self, label: &str) -> &mut Self {
        self.sub_apps.main.remove_update_system_by_label(label);
        self
    }

    /// Enables printing the organized schedule to stdout during setup.
    /// Useful for debugging system ordering.
    pub fn enable_schedule_print(&mut self) -> &mut Self {
        self.sub_apps.main.enable_schedule_print();
        self
    }

    /// Sets human-readable stage names for multi-stage simulations.
    pub fn set_stage_names(&mut self, names: &[&str]) -> &mut Self {
        self.sub_apps.main.set_stage_names(names);
        self
    }

    /// Registers a callback that produces domain-specific schedule warnings.
    pub fn set_warning_fn(&mut self, f: impl Fn(&[&str]) -> Vec<String> + 'static) -> &mut Self {
        self.sub_apps.main.set_warning_fn(f);
        self
    }
}

/// Internal error type for plugin registration failures.
#[derive(Debug)]
pub(crate) enum AppError {
    /// A unique plugin was registered more than once.
    DuplicatePlugin { plugin_name: String },
    /// One or more required dependency plugins (by [`TypeId`]) have not been registered yet.
    MissingDependencies {
        plugin_name: String,
        missing: Vec<TypeId>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{type_ids, Plugin};

    struct PluginA;
    impl Plugin for PluginA {
        fn build(&self, _app: &mut App) {}
        fn provides(&self) -> Vec<&str> {
            vec!["feature_a"]
        }
    }

    struct PluginB;
    impl Plugin for PluginB {
        fn build(&self, _app: &mut App) {}
        fn dependencies(&self) -> Vec<TypeId> {
            type_ids![PluginA]
        }
        fn requires(&self) -> Vec<&str> {
            vec!["feature_a"]
        }
    }

    struct PluginC;
    impl Plugin for PluginC {
        fn build(&self, _app: &mut App) {}
        fn requires(&self) -> Vec<&str> {
            vec!["feature_missing"]
        }
    }

    #[test]
    fn satisfied_dependencies_and_capabilities() {
        let mut app = App::new();
        app.add_plugins(PluginA);
        app.add_plugins(PluginB);
        // If we got here without panic, deps + capabilities are satisfied at build time.
        // Validate capability contracts explicitly.
        app.validate_capability_contracts(); // should not panic
    }

    #[test]
    #[should_panic(expected = "Missing plugin dependencies")]
    fn missing_typeid_dependency_panics() {
        let mut app = App::new();
        // PluginB depends on PluginA which is not registered.
        app.add_plugins(PluginB);
    }

    #[test]
    #[should_panic(expected = "Missing capabilities")]
    fn missing_capability_panics() {
        let mut app = App::new();
        app.add_plugins(PluginC);
        app.validate_capability_contracts();
    }

    #[test]
    fn type_ids_macro_produces_correct_ids() {
        let ids = type_ids![PluginA, PluginB];
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], TypeId::of::<PluginA>());
        assert_eq!(ids[1], TypeId::of::<PluginB>());
    }
}

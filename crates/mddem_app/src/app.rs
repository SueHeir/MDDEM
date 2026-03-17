//! The central [`App`] container and its supporting types.
//!
//! [`App`] is the entry point for every MDDEM simulation. It owns the main
//! [`SubApp`], coordinates plugin registration, and drives the simulation
//! lifecycle (setup → run → cleanup).
//!
//! # Typical usage
//!
//! ```rust,ignore
//! use mddem_app::prelude::*;
//!
//! App::new()
//!     .add_plugins(MyPlugins)
//!     .start();
//! ```

use std::{
    any::{Any, TypeId},
    cell::RefCell,
};

use mddem_scheduler::{IntoScheduledSystem, IntoSystem, ScheduleSet, ScheduleSetupSet};

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

        // Record the plugin name *before* build so that nested add_plugins calls
        // within build() can see this plugin as registered (prevents false-positive
        // dependency errors when a plugin group adds a dependency and its dependent
        // in sequence).
        self.main_mut()
            .plugin_names
            .insert(plugin.name().to_string());

        plugin.build(self);

        self.collect_config_snippet(&*plugin);

        Ok(self)
    }

    /// Checks that all plugins listed in `plugin.dependencies()` have already
    /// been registered. Returns `Err(AppError::MissingDependencies)` if any
    /// are missing.
    fn validate_dependencies(&self, plugin: &dyn Plugin) -> Result<(), AppError> {
        let deps = plugin.dependencies();
        if deps.is_empty() {
            return Ok(());
        }

        let missing: Vec<&str> = deps
            .iter()
            .filter(|dep| {
                // Check if any registered plugin name contains the dependency string.
                // This allows matching both full paths ("dem_atom::DemAtomPlugin")
                // and short names ("DemAtomPlugin").
                !self
                    .main()
                    .plugin_names
                    .iter()
                    .any(|name| name.contains(*dep) || dep.contains(name.as_str()))
            })
            .copied()
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(AppError::MissingDependencies {
                plugin_name: plugin.name().to_string(),
                missing: missing.iter().map(|s| s.to_string()).collect(),
            })
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
                .expect("ConfigSnippets resource has wrong type — this is a bug in MDDEM");
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

    /// Registers a system to run during the setup phase at the given [`ScheduleSetupSet`].
    pub fn add_setup_system<M>(
        &mut self,
        system: impl IntoScheduledSystem<M>,
        schedule_set: ScheduleSetupSet,
    ) -> &mut Self {
        self.sub_apps.main.add_setup_system(system, schedule_set);
        self
    }

    /// Registers a system to run every timestep at the given [`ScheduleSet`].
    pub fn add_update_system<M>(
        &mut self,
        system: impl IntoScheduledSystem<M>,
        schedule_set: ScheduleSet,
    ) -> &mut Self {
        self.sub_apps.main.add_update_system(system, schedule_set);
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
        self.sub_apps.main.start();
        self.run_cleanup();
    }

    /// Prints accumulated config snippets from all registered plugins.
    fn print_generated_config(&self) {
        let Some(snippets) = self.get_resource_ref::<ConfigSnippets>() else {
            return;
        };
        println!("# MDDEM generated configuration");
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
    pub fn remove_update_system<I, S: mddem_scheduler::System + 'static>(
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
}

/// Internal error type for plugin registration failures.
#[derive(Debug)]
pub(crate) enum AppError {
    /// A unique plugin was registered more than once.
    DuplicatePlugin { plugin_name: String },
    /// One or more required dependency plugins have not been registered yet.
    MissingDependencies {
        plugin_name: String,
        missing: Vec<String>,
    },
}

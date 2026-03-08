use std::{
    any::{Any, TypeId},
    cell::RefCell,
};

use mddem_scheduler::{IntoScheduledSystem, IntoSystem, ScheduleSet, ScheduleSetupSet, System};

use crate::{Plugin, Plugins, SubApp, SubApps};

/// Collected TOML snippets from all plugins that implement `default_config()`.
pub struct ConfigSnippets {
    pub snippets: Vec<String>,
}

/// Marker resource: when present, `App::start()` prints config snippets and exits.
pub struct GenerateConfigFlag;

/// Central application container. Holds resources, systems, and plugins.
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
    /// Creates a new [`App`] with some default structure to enable core engine features.
    /// This is the preferred constructor for most use cases.
    pub fn new() -> App {
        Self {
            sub_apps: SubApps {
                main: SubApp::new(),
            },
            cleanup_fns: Vec::new(),
        }
    }

    pub fn add_plugins<M>(&mut self, plugins: impl Plugins<M>) -> &mut Self {
        plugins.add_to_app(self);
        self
    }

    pub(crate) fn add_boxed_plugin(
        &mut self,
        plugin: Box<dyn Plugin>,
    ) -> Result<&mut Self, AppError> {
        if plugin.is_unique() && self.main_mut().plugin_names.contains(plugin.name()) {
            return Err(AppError::DuplicatePlugin {
                plugin_name: plugin.name().to_string(),
            });
        }

        plugin.build(self);

        if let Some(snippet) = plugin.default_config() {
            let snippet = snippet.to_string();
            if let Some(cell) = self.get_mut_resource(TypeId::of::<ConfigSnippets>()) {
                let mut borrow = cell.borrow_mut();
                let snippets = borrow.downcast_mut::<ConfigSnippets>().unwrap();
                snippets.snippets.push(snippet);
            } else {
                self.add_resource(ConfigSnippets {
                    snippets: vec![snippet],
                });
            }
        }

        Ok(self)
    }

    /// Returns a reference to the main [`SubApp`].
    pub fn main(&self) -> &SubApp {
        &self.sub_apps.main
    }

    /// Returns a mutable reference to the main [`SubApp`].
    pub fn main_mut(&mut self) -> &mut SubApp {
        &mut self.sub_apps.main
    }

    pub fn organize_systems(&mut self) {
        self.sub_apps.main.organize_systems();
    }

    pub fn setup(&mut self) -> &mut Self {
        self.sub_apps.main.setup();
        self
    }
    pub fn run(&mut self) -> &mut Self {
        self.sub_apps.main.run();
        self
    }

    pub fn add_setup_system<I, S: System + 'static>(
        &mut self,
        system: impl IntoSystem<I, System = S>,
        schedule_set: ScheduleSetupSet,
    ) -> &mut Self {
        self.sub_apps.main.add_setup_system(system, schedule_set);
        self
    }

    pub fn add_update_system<M>(
        &mut self,
        system: impl IntoScheduledSystem<M>,
        schedule_set: ScheduleSet,
    ) -> &mut Self {
        self.sub_apps.main.add_update_system(system, schedule_set);
        self
    }

    pub fn add_resource<R: 'static>(&mut self, res: R) -> &mut Self {
        self.sub_apps.main.add_resource(res);
        self
    }

    pub fn get_mut_resource(&mut self, res: TypeId) -> Option<&RefCell<Box<dyn Any>>> {
        self.sub_apps.main.get_mut_resource(res)
    }

    pub fn get_resource_ref<R: 'static>(&self) -> Option<std::cell::Ref<'_, R>> {
        self.sub_apps.main.get_resource_ref::<R>()
    }

    pub fn add_cleanup(&mut self, f: fn()) -> &mut Self {
        self.cleanup_fns.push(Box::new(f));
        self
    }

    pub fn start(&mut self) {
        if self.get_resource_ref::<GenerateConfigFlag>().is_some() {
            if let Some(snippets) = self.get_resource_ref::<ConfigSnippets>() {
                println!("# MDDEM generated configuration");
                println!("# Default values for all registered plugins\n");
                for snippet in &snippets.snippets {
                    println!("{}", snippet.trim());
                    println!();
                }
            }
            for f in self.cleanup_fns.drain(..) {
                f();
            }
            return;
        }
        self.sub_apps.main.start();
        for f in self.cleanup_fns.drain(..) {
            f();
        }
    }

    pub fn enable_schedule_print(&mut self) -> &mut Self {
        self.sub_apps.main.enable_schedule_print();
        self
    }
}

#[derive(Debug)]
pub(crate) enum AppError {
    DuplicatePlugin { plugin_name: String },
}

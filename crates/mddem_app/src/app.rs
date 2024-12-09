use std::collections::HashMap;

use mddem_scheduler::{IntoSystem, ScheduleSet, System};

use crate::{Plugin, Plugins, SubApp, SubApps};

pub struct App {
    pub(crate) sub_apps: SubApps,
}

// impl Debug for App {
//     fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
//         write!(f, "App {{ sub_apps: ")?;
//         f.debug_map()
//             .entries(self.sub_apps.sub_apps.iter())
//             .finish()?;
//         write!(f, "}}")
//     }
// }

impl Default for App {
    fn default() -> Self {
        let app = App::empty();
        app
    }
}

impl App {
    /// Creates a new [`App`] with some default structure to enable core engine features.
    /// This is the preferred constructor for most use cases.
    pub fn new() -> App {
        App::default()
    }

    /// Creates a new empty [`App`] with minimal default configuration.
    ///
    /// Use this constructor if you want to customize scheduling, exit handling, cleanup, etc.
    pub fn empty() -> App {
        Self {
            sub_apps: SubApps {
                main: SubApp::new(),
                sub_apps: HashMap::new(),
            },
        }
    }

    pub fn add_plugins<M>(&mut self, plugins: impl Plugins<M>) -> &mut Self {
        // if matches!(
        //     self.plugins_state(),
        //     PluginsState::Cleaned | PluginsState::Finished
        // ) {
        //     panic!(
        //         "Plugins cannot be added after App::cleanup() or App::finish() has been called."
        //     );
        // }
        plugins.add_to_app(self);
        self
    }

    pub(crate) fn add_boxed_plugin(
        &mut self,
        plugin: Box<dyn Plugin>,
    ) -> Result<&mut Self, AppError> {
        if plugin.is_unique() && self.main_mut().plugin_names.contains(plugin.name()) {
            return Err(AppError::DuplicatePlugin { plugin_name: plugin.name().to_string() })
        }

        // Reserve position in the plugin registry. If the plugin adds more plugins,
        // they'll all end up in insertion order.
        // let index = self.main().plugin_registry.len();
        // let name = self.main_mut().plugin_registry[index].name().to_string();
        plugin.build(self);
        self.main_mut()
            .plugin_registry
            .push(plugin);
        self.main_mut().plugin_build_depth += 1;
        // let result = catch_unwind(AssertUnwindSafe(|| plugin.build(self)));
        // self.main_mut()
        //     .plugin_names
        //     .insert(name);
        // self.main_mut().plugin_build_depth -= 1;

        // if let Err(payload) = result {
        //     resume_unwind(payload);
        // }
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
 
     pub fn setup(&mut self)  -> &mut Self {
         self.sub_apps.main.setup();
         self
     }
     pub fn run(&mut self)  -> &mut Self{
        self.sub_apps.main.run();
        self
     }
 
     pub fn add_setup_system<I, S: System + 'static>(
         &mut self,
         system: impl IntoSystem<I, System = S>,
         schedule_set: ScheduleSet,
     ) -> &mut Self{
         self.sub_apps.main.add_setup_system(system, schedule_set);
         self
     }
 
     pub fn add_update_system<I, S: System + 'static>(
         &mut self,
         system: impl IntoSystem<I, System = S>,
         schedule_set: ScheduleSet,
     ) -> &mut Self{
         self.sub_apps.main.add_update_system(system, schedule_set);
         self
     }
 
     pub fn add_resource<R: 'static>(&mut self, res: R) -> &mut Self { 
         self.sub_apps.main.add_resource(res);
         self
     }

    pub fn start(&mut self) {
        self.sub_apps.main.start();
    }
}


pub(crate) enum AppError {
    DuplicatePlugin { plugin_name: String },
}
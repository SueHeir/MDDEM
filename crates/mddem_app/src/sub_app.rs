use std::{any::{Any, TypeId}, cell::{RefCell, RefMut}, collections::{HashMap, HashSet}};

use crate::Plugin;

use mddem_scheduler::{IntoSystem, ScheduleSet, ScheduleSetupSet, Scheduler, System};

pub struct SubApp {
    pub(crate) scheduler: Scheduler,
    /// List of plugins that have been added.
    pub(crate) plugin_registry: Vec<Box<dyn Plugin>>,
    /// The names of plugins that have been added to this app. (used to track duplicates and
    /// already-registered plugins)
    pub(crate) plugin_names: HashSet<String>,
    /// Panics if an update is attempted while plugins are building.
    pub(crate) plugin_build_depth: usize,
}

// impl Debug for SubApp {
//     fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
//         write!(f, "SubApp")
//     }
// }

impl Default for SubApp {
    fn default() -> Self {
        let scheduler = Scheduler::default();
        Self {
            scheduler,
            plugin_registry: Vec::default(),
            plugin_names: HashSet::default(),
            plugin_build_depth: 0,
        }
    }


}


impl SubApp {
    /// Returns a default, empty [`SubApp`].
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(&mut self){
        self.scheduler.start();
    }

    pub fn organize_systems(&mut self) {
       self.scheduler.organize_systems();
    }

    pub fn setup(&mut self) {
        self.scheduler.setup();
    }
    pub fn run(&mut self) {
       self.scheduler.run();
    }

    pub fn add_setup_system<I, S: System + 'static>(
        &mut self,
        system: impl IntoSystem<I, System = S>,
        schedule_set: ScheduleSetupSet,
    ) {
        self.scheduler.add_setup_system(system, schedule_set);
    }

    pub fn add_update_system<I, S: System + 'static>(
        &mut self,
        system: impl IntoSystem<I, System = S>,
        schedule_set: ScheduleSet,
    ) {
        self.scheduler.add_update_system(system, schedule_set);
    }

    pub fn add_resource<R: 'static>(&mut self, res: R) { 
        self.scheduler.add_resource(res);
    }

    pub fn get_mut_resource(&mut self, res: TypeId) -> Option<&RefCell<Box<dyn Any>>> {
        self.scheduler.get_mut_resource(res)
    }

}


/// The collection of sub-apps that belong to an [`App`].
#[derive(Default)]
pub struct SubApps {
    /// The primary sub-app that contains the "main" world.
    pub main: SubApp,
    /// Other, labeled sub-apps.
    pub sub_apps: HashMap<String, SubApp>,
}

impl SubApps { 

}

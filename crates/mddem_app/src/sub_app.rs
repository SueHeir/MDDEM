use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::HashSet,
};

use mddem_scheduler::{
    IntoScheduledSystem, IntoSystem, ScheduleSet, ScheduleSetupSet, Scheduler, System,
};

/// Holds one [`Scheduler`] instance with its resource store and system lists.
pub struct SubApp {
    pub(crate) scheduler: Scheduler,
    /// The names of plugins that have been added to this app. (used to track duplicates and
    /// already-registered plugins)
    pub(crate) plugin_names: HashSet<String>,
}

impl Default for SubApp {
    fn default() -> Self {
        Self {
            scheduler: Scheduler::default(),
            plugin_names: HashSet::default(),
        }
    }
}

impl SubApp {
    /// Returns a default, empty [`SubApp`].
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(&mut self) {
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

    pub fn add_update_system<M>(
        &mut self,
        system: impl IntoScheduledSystem<M>,
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

    pub fn get_resource_ref<R: 'static>(&self) -> Option<std::cell::Ref<'_, R>> {
        self.scheduler.get_resource_ref::<R>()
    }

    pub fn remove_update_system<I, S: mddem_scheduler::System + 'static>(
        &mut self,
        system: impl IntoSystem<I, System = S>,
    ) {
        self.scheduler.remove_update_system(system);
    }

    pub fn remove_update_system_by_label(&mut self, label: &str) {
        self.scheduler.remove_update_system_by_label(label);
    }

    pub fn enable_schedule_print(&mut self) {
        self.scheduler.enable_schedule_print();
    }
}

/// The collection of sub-apps that belong to an [`App`].
#[derive(Default)]
pub struct SubApps {
    /// The primary sub-app that contains the "main" world.
    pub main: SubApp,
}

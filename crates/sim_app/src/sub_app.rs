//! The [`SubApp`] type — a self-contained scheduler with its own resource store.
//!
//! Currently every [`App`](crate::App) has exactly one `SubApp` (the "main"
//! sub-app). The abstraction exists to support future multi-world scenarios
//! (e.g. rendering in a separate sub-app).

use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::HashSet,
};

use sim_scheduler::{
    IntoScheduledSystem, IntoSystem, SchedulePhase, Scheduler,
};

/// A self-contained simulation world: one [`Scheduler`] with its resource store
/// and system lists.
///
/// Most users interact with `SubApp` indirectly through [`App`](crate::App),
/// which delegates to its main `SubApp`.
#[derive(Default)]
pub struct SubApp {
    pub(crate) scheduler: Scheduler,
    /// The names of plugins that have been added to this sub-app (used to track
    /// duplicates and already-registered plugins).
    pub(crate) plugin_names: HashSet<String>,
}


impl SubApp {
    /// Creates a new, empty [`SubApp`] with default scheduler settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Runs the full simulation lifecycle: organize → setup → run.
    pub fn start(&mut self) {
        self.scheduler.start();
    }

    /// Sorts systems into their schedule-set execution order.
    pub fn organize_systems(&mut self) {
        self.scheduler.organize_systems();
    }

    /// Executes all setup-phase systems.
    pub fn setup(&mut self) {
        self.scheduler.setup();
    }

    /// Executes the main simulation loop (update systems each timestep).
    pub fn run(&mut self) {
        self.scheduler.run();
    }

    /// Registers a system to run during the setup phase at the given schedule phase.
    pub fn add_setup_system<M>(
        &mut self,
        system: impl IntoScheduledSystem<M>,
        schedule_set: impl SchedulePhase,
    ) {
        self.scheduler.add_setup_system(system, schedule_set);
    }

    /// Registers a system to run every timestep at the given schedule phase.
    pub fn add_update_system<M>(
        &mut self,
        system: impl IntoScheduledSystem<M>,
        schedule_set: impl SchedulePhase,
    ) {
        self.scheduler.add_update_system(system, schedule_set);
    }

    /// Assigns a namespace to all systems registered under the given phase enum type.
    pub fn set_phase_namespace<P: SchedulePhase + 'static>(&mut self, namespace: u32) {
        self.scheduler.set_phase_namespace::<P>(namespace);
    }

    /// Inserts a resource into this sub-app's resource store.
    pub fn add_resource<R: 'static>(&mut self, res: R) {
        self.scheduler.add_resource(res);
    }

    /// Returns a mutable reference to the raw resource cell for the given [`TypeId`].
    pub fn get_mut_resource(&mut self, res: TypeId) -> Option<&RefCell<Box<dyn Any>>> {
        self.scheduler.get_mut_resource(res)
    }

    /// Returns a borrowed reference to a resource of type `R`, or `None` if absent.
    pub fn get_resource_ref<R: 'static>(&self) -> Option<std::cell::Ref<'_, R>> {
        self.scheduler.get_resource_ref::<R>()
    }

    /// Removes an update system by its concrete type.
    pub fn remove_update_system<I, S: sim_scheduler::System + 'static>(
        &mut self,
        system: impl IntoSystem<I, System = S>,
    ) {
        self.scheduler.remove_update_system(system);
    }

    /// Removes an update system identified by its string label.
    pub fn remove_update_system_by_label(&mut self, label: &str) {
        self.scheduler.remove_update_system_by_label(label);
    }

    /// Enables printing the organized schedule to stdout during setup.
    pub fn enable_schedule_print(&mut self) {
        self.scheduler.enable_schedule_print();
    }

    /// Sets human-readable stage names for multi-stage simulations.
    pub fn set_stage_names(&mut self, names: &[&str]) {
        self.scheduler.set_stage_names(names);
    }

    /// Registers a callback that produces domain-specific schedule warnings.
    pub fn set_warning_fn(&mut self, f: impl Fn(&[&str]) -> Vec<String> + 'static) {
        self.scheduler.set_warning_fn(f);
    }
}

/// The collection of sub-apps that belong to an [`App`](crate::App).
#[derive(Default)]
pub struct SubApps {
    /// The primary sub-app that contains the "main" world.
    pub main: SubApp,
}

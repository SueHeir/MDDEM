//! The [`Plugin`] trait, [`PluginGroup`] builder, and state/stage management plugins.
//!
//! Every feature is implemented as a [`Plugin`]. Plugins register
//! resources, systems, and sub-plugins with the [`App`](crate::App) during
//! the build phase.
//!
//! # Implementing a plugin
//!
//! ```rust,ignore
//! use sim_app::prelude::*;
//! use sim_scheduler::ScheduleSet;
//!
//! pub struct GravityPlugin;
//!
//! impl Plugin for GravityPlugin {
//!     fn build(&self, app: &mut App) {
//!         app.add_update_system(apply_gravity, ScheduleSet::PreForce);
//!     }
//! }
//! ```

use downcast_rs::{impl_downcast, Downcast};
use sim_scheduler::{
    apply_state_transitions, check_stage_advance, CurrentState, NextState, SchedulePhase,
    StageName, StoredPhase,
};
use std::marker::PhantomData;

use crate::App;
use core::any::Any;
use std::any::TypeId;
use std::collections::HashSet;

/// A self-contained module that registers resources and systems with an [`App`].
///
/// Every simulation feature — physics models, I/O, analysis — is implemented
/// as a `Plugin`. The [`build`](Plugin::build) method is called once during
/// [`App::add_plugins`] to wire everything up.
///
/// # Example
///
/// ```rust,ignore
/// pub struct MyPlugin;
///
/// impl Plugin for MyPlugin {
///     fn build(&self, app: &mut App) {
///         app.add_resource(MyResource::default())
///            .add_update_system(my_system, ScheduleSet::Update);
///     }
/// }
/// ```
///
/// Closures `Fn(&mut App)` also implement `Plugin`, which is handy for tests:
///
/// ```rust,ignore
/// app.add_plugins(|app: &mut App| {
///     app.add_resource(TestResource);
/// });
/// ```
pub trait Plugin: Downcast + Any + Send + Sync {
    /// Configures the [`App`] to which this plugin is added.
    ///
    /// This is the main entry point for plugin setup. Register resources,
    /// systems, and sub-plugins here.
    fn build(&self, app: &mut App);

    /// Returns the plugin's name, used for duplicate detection and diagnostics.
    ///
    /// Defaults to the Rust type name (e.g. `"my_crate::MyPlugin"`).
    fn name(&self) -> &str {
        core::any::type_name::<Self>()
    }

    /// Whether this plugin may only be added once.
    ///
    /// Returns `true` by default. Override to return `false` if the plugin can
    /// be meaningfully instantiated multiple times (e.g. with different configs).
    fn is_unique(&self) -> bool {
        true
    }

    /// Returns a TOML snippet showing this plugin's config section with defaults.
    ///
    /// Used by `--generate-config` to print a complete example config file.
    /// Return `None` (the default) if the plugin has no configuration.
    fn default_config(&self) -> Option<&str> {
        None
    }

    /// Returns the names of plugins that must be registered before this one.
    ///
    /// The app validates that all listed plugins have been registered before
    /// this plugin's [`build`](Plugin::build) runs, and produces a clear error
    /// message if any are missing.
    ///
    /// ```rust,ignore
    /// fn dependencies(&self) -> Vec<&str> {
    ///     vec!["dem_wall::WallPlugin", "dem_atom::DemAtomPlugin"]
    /// }
    /// ```
    fn dependencies(&self) -> Vec<&str> {
        Vec::new()
    }
}

impl_downcast!(Plugin);

/// Blanket impl: any `Fn(&mut App) + Send + Sync + 'static` can be used as a
/// [`Plugin`]. This is convenient for quick inline registrations and tests.
impl<T: Fn(&mut App) + Send + Sync + 'static> Plugin for T {
    fn build(&self, app: &mut App) {
        self(app);
    }
}

// ─── PluginGroup ──────────────────────────────────────────────────────────────

/// A collection of plugins that can be added to an [`App`] as a single unit.
///
/// Implement this to bundle multiple plugins together for reuse:
///
/// ```rust,ignore
/// pub struct MyDefaultPlugins;
///
/// impl PluginGroup for MyDefaultPlugins {
///     fn build(self) -> PluginGroupBuilder {
///         PluginGroupBuilder::start::<Self>()
///             .add(FooPlugin)
///             .add(BarPlugin)
///     }
/// }
///
/// App::new().add_plugins(MyDefaultPlugins).start();
/// ```
pub trait PluginGroup: Sized {
    /// Constructs a [`PluginGroupBuilder`] containing the group's plugins.
    fn build(self) -> PluginGroupBuilder;
}

/// Builder for a [`PluginGroup`]. Add plugins in order; they will be registered
/// in that order when the group is applied to an [`App`].
pub struct PluginGroupBuilder {
    plugins: Vec<Box<dyn Plugin>>,
    disabled: HashSet<TypeId>,
}

impl PluginGroupBuilder {
    /// Creates a new, empty builder for plugin group `G`.
    pub fn start<G: PluginGroup>() -> Self {
        Self {
            plugins: Vec::new(),
            disabled: HashSet::new(),
        }
    }

    /// Adds a plugin to the group. Plugins that have been [`disable`](Self::disable)d
    /// are silently skipped.
    #[allow(clippy::should_implement_trait)]
    pub fn add<P: Plugin>(mut self, plugin: P) -> Self {
        if self.disabled.contains(&TypeId::of::<P>()) {
            return self;
        }
        self.plugins.push(Box::new(plugin));
        self
    }

    /// Marks a plugin type as disabled so that subsequent [`add`](Self::add) calls
    /// for that type are ignored.
    pub fn disable<P: Plugin>(mut self) -> Self {
        self.disabled.insert(TypeId::of::<P>());
        self
    }

    /// Registers all collected plugins with the given [`App`].
    ///
    /// # Panics
    ///
    /// Panics if any plugin fails registration (duplicate or missing dependencies).
    pub(crate) fn finish(self, app: &mut App) {
        for plugin in self.plugins {
            app.add_boxed_plugin(plugin)
                .expect("failed to add plugin from PluginGroup — check for duplicates or missing dependencies");
        }
    }
}

// ─── StatesPlugin ─────────────────────────────────────────────────────────────

/// Registers [`CurrentState<S>`] and [`NextState<S>`] resources and wires up the
/// end-of-step transition system at the given schedule phase.
///
/// ```rust,ignore
/// #[derive(Clone, PartialEq, Default)]
/// enum Phase { #[default] Settling, Production }
///
/// App::new()
///     .add_plugins(StatesPlugin::new(Phase::Settling, ScheduleSet::PostFinalIntegration))
///     ...
/// ```
pub struct StatesPlugin<S: Clone + PartialEq + Default + Send + Sync + 'static> {
    /// The initial state value to use at simulation start.
    pub initial: S,
    /// The schedule phase at which state transitions are applied.
    phase: StoredPhase,
}

impl<S: Clone + PartialEq + Default + Send + Sync + 'static> StatesPlugin<S> {
    /// Creates a new [`StatesPlugin`] with the given initial state and schedule phase.
    pub fn new(initial: S, phase: impl SchedulePhase) -> Self {
        Self {
            initial,
            phase: StoredPhase::from(phase),
        }
    }
}

impl<S: Clone + PartialEq + Default + Send + Sync + 'static> Plugin for StatesPlugin<S> {
    fn build(&self, app: &mut App) {
        app.add_resource(CurrentState(self.initial.clone()));
        app.add_resource(NextState::<S>(None));
        app.add_update_system(
            apply_state_transitions::<S>,
            self.phase,
        );
    }
}

// ─── StageAdvancePlugin ──────────────────────────────────────────────────────

/// Watches for [`CurrentState<S>`] changes and sets `SchedulerManager::advance_requested`.
///
/// Add alongside [`StatesPlugin`] when using `#[derive(StageEnum)]`:
///
/// ```rust,ignore
/// app.add_plugins(StatesPlugin::new(Phase::Settle, ScheduleSet::PostFinalIntegration));
/// app.add_plugins(StageAdvancePlugin::<Phase>::new(ScheduleSet::PostFinalIntegration));
/// ```
pub struct StageAdvancePlugin<S: StageName + Clone + PartialEq + Default + Send + Sync + 'static> {
    _marker: PhantomData<S>,
    phase: StoredPhase,
}

impl<S: StageName + Clone + PartialEq + Default + Send + Sync + 'static> StageAdvancePlugin<S> {
    /// Creates a new [`StageAdvancePlugin`] for state type `S` at the given schedule phase.
    pub fn new(phase: impl SchedulePhase) -> Self {
        Self {
            _marker: PhantomData,
            phase: StoredPhase::from(phase),
        }
    }
}

impl<S: StageName + Clone + PartialEq + Default + Send + Sync + 'static> Plugin
    for StageAdvancePlugin<S>
{
    fn build(&self, app: &mut App) {
        // Store stage names for validation and DOT export
        app.add_resource(StageNames(S::stage_names()));
        app.set_stage_names(S::stage_names());
        app.add_update_system(
            check_stage_advance::<S>,
            self.phase,
        );
    }
}

/// Resource storing stage name strings, used for validation and DOT export.
pub struct StageNames(pub &'static [&'static str]);

// ─── Plugins sealed trait ─────────────────────────────────────────────────────

/// Types that represent a set of [`Plugin`]s.
///
/// This trait is implemented for:
/// - Individual types implementing [`Plugin`]
/// - Types implementing [`PluginGroup`]
/// - Tuples of [`Plugins`]
///
/// You do not need to implement this trait yourself.
pub trait Plugins<Marker>: sealed::Plugins<Marker> {}
impl<Marker, T> Plugins<Marker> for T where T: sealed::Plugins<Marker> {}

pub(crate) mod sealed {
    use crate::app::AppError;
    use crate::{App, Plugin, PluginGroup};

    pub trait Plugins<Marker> {
        fn add_to_app(self, app: &mut App);
    }

    pub struct PluginMarker;
    pub struct PluginGroupMarker;

    impl<P: Plugin> Plugins<PluginMarker> for P {
        #[track_caller]
        fn add_to_app(self, app: &mut App) {
            match app.add_boxed_plugin(Box::new(self)) {
                Err(AppError::DuplicatePlugin { plugin_name }) => {
                    panic!(
                        "Error adding plugin {plugin_name}: plugin was already added in application"
                    )
                }
                Err(AppError::MissingDependencies {
                    plugin_name,
                    missing,
                }) => {
                    eprintln!();
                    eprintln!("ERROR: Plugin `{}` is missing required dependencies:", plugin_name);
                    for dep in &missing {
                        eprintln!("  - {}", dep);
                    }
                    eprintln!();
                    eprintln!("  Hint: Add the missing plugin(s) before `{}`.", plugin_name);
                    eprintln!("  Example plugin registration order:");
                    for dep in &missing {
                        eprintln!("    app.add_plugins({});", dep.rsplit("::").next().unwrap_or(dep));
                    }
                    eprintln!("    app.add_plugins({});", plugin_name.rsplit("::").next().unwrap_or(&plugin_name));
                    panic!(
                        "Missing plugin dependencies for `{}`: {:?}",
                        plugin_name, missing
                    );
                }
                Ok(_) => {}
            }
        }
    }

    impl<G: PluginGroup> Plugins<PluginGroupMarker> for G {
        fn add_to_app(self, app: &mut App) {
            self.build().finish(app);
        }
    }
}

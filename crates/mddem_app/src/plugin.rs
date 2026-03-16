use downcast_rs::{impl_downcast, Downcast};
use mddem_scheduler::{
    apply_state_transitions, check_stage_advance, CurrentState, NextState, ScheduleSet, StageName,
};
use std::marker::PhantomData;

use crate::App;
use core::any::Any;
use std::any::TypeId;
use std::collections::HashSet;

/// A self-contained module that registers resources and systems with an [`App`].
pub trait Plugin: Downcast + Any + Send + Sync {
    /// Configures the [`App`] to which this plugin is added.
    fn build(&self, app: &mut App);

    fn name(&self) -> &str {
        core::any::type_name::<Self>()
    }

    /// If the plugin can be meaningfully instantiated several times in an [`App`],
    /// override this method to return `false`.
    fn is_unique(&self) -> bool {
        true
    }

    /// Return a TOML snippet showing this plugin's config section with defaults.
    /// Used by `--generate-config` to print a complete example config file.
    fn default_config(&self) -> Option<&str> {
        None
    }

    /// Return the names of plugins that must be registered before this one.
    ///
    /// Override this to declare dependencies. The app will validate that all
    /// listed plugins have been registered before this plugin's `build()` runs,
    /// and produce a clear error message if any are missing.
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
    fn build(self) -> PluginGroupBuilder;
}

/// Builder for a [`PluginGroup`]. Add plugins in order; they will be registered in that order.
pub struct PluginGroupBuilder {
    plugins: Vec<Box<dyn Plugin>>,
    disabled: HashSet<TypeId>,
}

impl PluginGroupBuilder {
    pub fn start<G: PluginGroup>() -> Self {
        Self {
            plugins: Vec::new(),
            disabled: HashSet::new(),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add<P: Plugin>(mut self, plugin: P) -> Self {
        if self.disabled.contains(&TypeId::of::<P>()) {
            return self;
        }
        self.plugins.push(Box::new(plugin));
        self
    }

    pub fn disable<P: Plugin>(mut self) -> Self {
        self.disabled.insert(TypeId::of::<P>());
        self
    }

    pub(crate) fn finish(self, app: &mut App) {
        for plugin in self.plugins {
            app.add_boxed_plugin(plugin).unwrap();
        }
    }
}

// ─── StatesPlugin ─────────────────────────────────────────────────────────────

/// Registers `CurrentState<S>` and `NextState<S>` resources and wires up the
/// end-of-step transition system. Transitions are applied at `PostFinalIntegration`.
///
/// ```rust,ignore
/// #[derive(Clone, PartialEq, Default)]
/// enum Phase { #[default] Settling, Production }
///
/// App::new()
///     .add_plugins(StatesPlugin { initial: Phase::Settling })
///     ...
/// ```
pub struct StatesPlugin<S: Clone + PartialEq + Default + Send + Sync + 'static> {
    pub initial: S,
}

impl<S: Clone + PartialEq + Default + Send + Sync + 'static> Plugin for StatesPlugin<S> {
    fn build(&self, app: &mut App) {
        app.add_resource(CurrentState(self.initial.clone()));
        app.add_resource(NextState::<S>(None));
        app.add_update_system(
            apply_state_transitions::<S>,
            ScheduleSet::PostFinalIntegration,
        );
    }
}

// ─── StageAdvancePlugin ──────────────────────────────────────────────────────

/// Watches for `CurrentState<S>` changes and sets `SchedulerManager::advance_requested`.
///
/// Add alongside `StatesPlugin` when using `#[derive(StageEnum)]`:
/// ```rust,ignore
/// app.add_plugins(StatesPlugin { initial: Phase::Settle });
/// app.add_plugins(StageAdvancePlugin::<Phase>::new());
/// ```
pub struct StageAdvancePlugin<S: StageName + Clone + PartialEq + Default + Send + Sync + 'static> {
    _marker: PhantomData<S>,
}

impl<S: StageName + Clone + PartialEq + Default + Send + Sync + 'static> StageAdvancePlugin<S> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
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
            ScheduleSet::PostFinalIntegration,
        );
    }
}

/// Resource storing stage name strings for validation.
pub struct StageNames(pub &'static [&'static str]);

// ─── Plugins sealed trait ─────────────────────────────────────────────────────

/// Types that represent a set of [`Plugin`]s.
///
/// Implemented for all types which implement [`Plugin`], [`PluginGroup`], and
/// tuples over [`Plugins`].
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

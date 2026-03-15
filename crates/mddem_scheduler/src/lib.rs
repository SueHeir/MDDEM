//! Dependency-injection scheduler with [`Res`]/[`ResMut`] resource access and ordered system execution.
//!
//! Systems are plain functions whose parameters implement [`SystemParam`]. The scheduler resolves
//! resource indices at startup and executes systems in [`ScheduleSet`] order each timestep.

#![allow(clippy::too_many_arguments)]
// ANCHOR: All
use std::any::{Any, TypeId};
use std::cell::{Ref, RefCell, RefMut};
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

// ─── System macro ─────────────────────────────────────────────────────────────

macro_rules! impl_system {
    ($($params:ident),*) => {
        #[allow(non_snake_case, unused)]
        impl<F, $($params: SystemParam),*> System for FunctionSystem<($($params,)*), F>
            where
                for<'a, 'b> &'a mut F:
                    FnMut($($params),*) +
                    FnMut($(<$params as SystemParam>::Item<'b>),*)
        {
            fn run(&mut self, resources: &[RefCell<Box<dyn Any>>]) {
                fn call_inner<$($params),*>(
                    mut f: impl FnMut($($params),*),
                    $($params: $params),*
                ) { f($($params),*) }

                let locals_ptr = &mut self.locals as *mut _;
                let mut _param_idx = 0usize;
                $(
                    let $params = $params::retrieve(resources, self.indices[_param_idx], locals_ptr);
                    _param_idx += 1;
                )*
                call_inner(&mut self.f, $($params),*)
            }

            fn prepare(&mut self, index: &HashMap<TypeId, usize>) -> Vec<String> {
                self.indices.clear();
                let mut _missing = Vec::new();
                $(
                    let _type_info = <$params as SystemParam>::resource_type_id();
                    let _idx = _type_info
                        .and_then(|(tid, _)| index.get(&tid).copied())
                        .unwrap_or(usize::MAX);
                    if _idx == usize::MAX && !<$params as SystemParam>::is_optional() {
                        if let Some((_, name)) = _type_info {
                            _missing.push(name.to_string());
                        }
                    }
                    self.indices.push(_idx);
                )*
                _missing
            }

            fn name(&self) -> &str { std::any::type_name::<F>() }
        }
    }
}

macro_rules! impl_into_system {
    ($($params:ident),*) => {
        impl<F, $($params: SystemParam),*> IntoSystem<($($params,)*)> for F
            where
                for<'a, 'b> &'a mut F:
                    FnMut($($params),*) +
                    FnMut($(<$params as SystemParam>::Item<'b>),*)
        {
            type System = FunctionSystem<($($params,)*), Self>;
            fn into_system(self) -> Self::System {
                FunctionSystem { f: self, marker: Default::default(), locals: HashMap::new(), indices: Vec::new() }
            }
        }
    }
}

// ─── Condition macro ──────────────────────────────────────────────────────────

macro_rules! impl_condition {
    ($($params:ident),*) => {
        #[allow(non_snake_case, unused)]
        impl<F, $($params: SystemParam),*> Condition for FunctionCondition<($($params,)*), F>
            where
                for<'a, 'b> &'a mut F:
                    FnMut($($params),*) -> bool +
                    FnMut($(<$params as SystemParam>::Item<'b>),*) -> bool
        {
            fn evaluate(&mut self, resources: &[RefCell<Box<dyn Any>>]) -> bool {
                fn call_inner<$($params),*>(
                    mut f: impl FnMut($($params),*) -> bool,
                    $($params: $params),*
                ) -> bool { f($($params),*) }

                let locals_ptr = &mut self.locals as *mut _;
                let mut _param_idx = 0usize;
                $(
                    let $params = $params::retrieve(resources, self.indices[_param_idx], locals_ptr);
                    _param_idx += 1;
                )*
                call_inner(&mut self.f, $($params),*)
            }

            fn prepare(&mut self, index: &HashMap<TypeId, usize>) -> Vec<String> {
                self.indices.clear();
                let mut _missing = Vec::new();
                $(
                    let _type_info = <$params as SystemParam>::resource_type_id();
                    let _idx = _type_info
                        .and_then(|(tid, _)| index.get(&tid).copied())
                        .unwrap_or(usize::MAX);
                    if _idx == usize::MAX && !<$params as SystemParam>::is_optional() {
                        if let Some((_, name)) = _type_info {
                            _missing.push(name.to_string());
                        }
                    }
                    self.indices.push(_idx);
                )*
                _missing
            }

            fn name(&self) -> &str {
                std::any::type_name::<F>()
            }
        }
    }
}

macro_rules! impl_into_condition {
    ($($params:ident),*) => {
        impl<F, $($params: SystemParam),*> IntoCondition<($($params,)*)> for F
            where
                for<'a, 'b> &'a mut F:
                    FnMut($($params),*) -> bool +
                    FnMut($(<$params as SystemParam>::Item<'b>),*) -> bool
        {
            type Condition = FunctionCondition<($($params,)*), Self>;
            fn into_condition(self) -> Self::Condition {
                FunctionCondition { f: self, marker: Default::default(), locals: HashMap::new(), indices: Vec::new() }
            }
        }
    }
}

// ─── SystemParam ──────────────────────────────────────────────────────────────

// ANCHOR: SystemParam
/// Types that can be injected as parameters into system functions.
pub trait SystemParam {
    type Item<'new>;
    fn retrieve<'r>(
        resources: &'r [RefCell<Box<dyn Any>>],
        index: usize,
        locals: *mut HashMap<TypeId, Box<dyn Any>>,
    ) -> Self::Item<'r>;
    fn resource_type_id() -> Option<(TypeId, &'static str)> {
        None
    }
    fn is_optional() -> bool {
        false
    }
}
// ANCHOR_END: SystemParam

// ANCHOR: ResSystemParam
impl<'res, T: 'static> SystemParam for Res<'res, T> {
    type Item<'new> = Res<'new, T>;
    fn retrieve<'r>(
        resources: &'r [RefCell<Box<dyn Any>>],
        index: usize,
        _locals: *mut HashMap<TypeId, Box<dyn Any>>,
    ) -> Self::Item<'r> {
        let guard = resources[index].borrow();
        // Downcast once here; Deref uses the cached pointer.
        let ptr: *const T = guard.downcast_ref::<T>().unwrap();
        Res { _guard: guard, ptr }
    }
    fn resource_type_id() -> Option<(TypeId, &'static str)> {
        Some((TypeId::of::<T>(), std::any::type_name::<T>()))
    }
}
// ANCHOR_END: ResSystemParam

// ANCHOR: ResMutSystemParam
impl<'res, T: 'static> SystemParam for ResMut<'res, T> {
    type Item<'new> = ResMut<'new, T>;
    fn retrieve<'r>(
        resources: &'r [RefCell<Box<dyn Any>>],
        index: usize,
        _locals: *mut HashMap<TypeId, Box<dyn Any>>,
    ) -> Self::Item<'r> {
        let mut guard = resources[index].borrow_mut();
        // Downcast once here; Deref/DerefMut use the cached pointer.
        let ptr: *mut T = guard.downcast_mut::<T>().unwrap();
        ResMut { _guard: guard, ptr }
    }
    fn resource_type_id() -> Option<(TypeId, &'static str)> {
        Some((TypeId::of::<T>(), std::any::type_name::<T>()))
    }
}
// ANCHOR_END: ResMutSystemParam

// ─── Res / ResMut / Local ─────────────────────────────────────────────────────

// ANCHOR: Res
/// Shared immutable reference to resource `T`, injected into systems.
///
/// The downcast from `Box<dyn Any>` happens once at construction (in `retrieve`).
/// `Deref` then uses the cached pointer — no virtual calls in hot loops.
pub struct Res<'a, T: 'static> {
    _guard: Ref<'a, Box<dyn Any>>,
    ptr: *const T,
}
// ANCHOR_END: Res

impl<T: 'static> Deref for Res<'_, T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        // SAFETY: ptr was obtained from downcast_ref in retrieve() and is valid
        // for the lifetime 'a, guaranteed by _guard holding the Ref borrow.
        unsafe { &*self.ptr }
    }
}

// ANCHOR: ResMut
/// Exclusive mutable reference to resource `T`, injected into systems.
///
/// The downcast from `Box<dyn Any>` happens once at construction (in `retrieve`).
/// `Deref`/`DerefMut` then use the cached pointer — no virtual calls in hot loops.
pub struct ResMut<'a, T: 'static> {
    _guard: RefMut<'a, Box<dyn Any>>,
    ptr: *mut T,
}

impl<T: 'static> Deref for ResMut<'_, T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        // SAFETY: ptr was obtained from downcast_mut in retrieve() and is valid
        // for the lifetime 'a, guaranteed by _guard holding the RefMut borrow.
        unsafe { &*self.ptr }
    }
}
impl<T: 'static> DerefMut for ResMut<'_, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: same as Deref; exclusive access guaranteed by RefMut.
        unsafe { &mut *self.ptr }
    }
}
// ANCHOR_END: ResMut

// ANCHOR: Local
/// Per-system local state. Persists across invocations of the same system instance.
/// Initialized with `T::default()` on first access.
pub struct Local<'a, T: Default + 'static> {
    value: &'a mut T,
    _marker: PhantomData<&'a mut T>,
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
impl<'res, T: Default + 'static> SystemParam for Local<'res, T> {
    type Item<'new> = Local<'new, T>;
    fn retrieve<'r>(
        _resources: &'r [RefCell<Box<dyn Any>>],
        _index: usize,
        locals: *mut HashMap<TypeId, Box<dyn Any>>,
    ) -> Self::Item<'r> {
        // SAFETY: locals points to FunctionSystem::locals, exclusively owned by this
        // system and alive for the duration of this retrieve call.
        let map = unsafe { &mut *locals };
        let entry = map
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(T::default()));
        Local {
            value: entry.downcast_mut::<T>().unwrap(),
            _marker: PhantomData,
        }
    }
}

impl<T: Default + 'static> Deref for Local<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value
    }
}
impl<T: Default + 'static> DerefMut for Local<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value
    }
}
// ANCHOR_END: Local

// ─── Option<Res<T>> / Option<ResMut<T>> ──────────────────────────────────────

impl<'res, T: 'static> SystemParam for Option<Res<'res, T>> {
    type Item<'new> = Option<Res<'new, T>>;
    fn retrieve<'r>(
        resources: &'r [RefCell<Box<dyn Any>>],
        index: usize,
        _locals: *mut HashMap<TypeId, Box<dyn Any>>,
    ) -> Self::Item<'r> {
        if index == usize::MAX {
            return None;
        }
        let guard = resources[index].borrow();
        let ptr: *const T = guard.downcast_ref::<T>().unwrap();
        Some(Res { _guard: guard, ptr })
    }
    fn resource_type_id() -> Option<(TypeId, &'static str)> {
        Some((TypeId::of::<T>(), std::any::type_name::<T>()))
    }
    fn is_optional() -> bool {
        true
    }
}

impl<'res, T: 'static> SystemParam for Option<ResMut<'res, T>> {
    type Item<'new> = Option<ResMut<'new, T>>;
    fn retrieve<'r>(
        resources: &'r [RefCell<Box<dyn Any>>],
        index: usize,
        _locals: *mut HashMap<TypeId, Box<dyn Any>>,
    ) -> Self::Item<'r> {
        if index == usize::MAX {
            return None;
        }
        let mut guard = resources[index].borrow_mut();
        let ptr: *mut T = guard.downcast_mut::<T>().unwrap();
        Some(ResMut { _guard: guard, ptr })
    }
    fn resource_type_id() -> Option<(TypeId, &'static str)> {
        Some((TypeId::of::<T>(), std::any::type_name::<T>()))
    }
    fn is_optional() -> bool {
        true
    }
}

// ─── System trait & FunctionSystem ───────────────────────────────────────────

// ANCHOR: System
/// A runnable unit of work that receives resources via dependency injection.
pub trait System {
    fn run(&mut self, resources: &[RefCell<Box<dyn Any>>]);
    fn prepare(&mut self, _index: &HashMap<TypeId, usize>) -> Vec<String> {
        Vec::new()
    }
    fn name(&self) -> &str {
        "unknown"
    }
    fn condition_name(&self) -> Option<&str> {
        None
    }
}
// ANCHOR_END: System

pub struct FunctionSystem<Input, F> {
    f: F,
    marker: PhantomData<fn() -> Input>,
    /// Per-system-instance local state, keyed by TypeId.
    locals: HashMap<TypeId, Box<dyn Any>>,
    /// Cached resource indices resolved during prepare().
    indices: Vec<usize>,
}

pub trait IntoSystem<Input> {
    type System: System;
    fn into_system(self) -> Self::System;
}

impl_system!();
impl_system!(T1);
impl_system!(T1, T2);
impl_system!(T1, T2, T3);
impl_system!(T1, T2, T3, T4);
impl_system!(T1, T2, T3, T4, T5);
impl_system!(T1, T2, T3, T4, T5, T6);
impl_system!(T1, T2, T3, T4, T5, T6, T7);
impl_system!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_system!(T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_system!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);

impl_into_system!();
impl_into_system!(T1);
impl_into_system!(T1, T2);
impl_into_system!(T1, T2, T3);
impl_into_system!(T1, T2, T3, T4);
impl_into_system!(T1, T2, T3, T4, T5);
impl_into_system!(T1, T2, T3, T4, T5, T6);
impl_into_system!(T1, T2, T3, T4, T5, T6, T7);
impl_into_system!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_into_system!(T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_into_system!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);

// ─── IntoSystemLabel ─────────────────────────────────────────────────────────

/// Marker for string-based labels (`&str`, `String`).
pub struct StrLabelMarker;
/// Marker for function-handle-based labels.
pub struct FnLabelMarker<I>(PhantomData<I>);

/// Converts a label source (string or function handle) into a `String` label.
///
/// Strings pass through directly. Function handles resolve to their
/// `std::any::type_name`, matching the system's registered name.
pub trait IntoSystemLabel<M> {
    fn into_label(self) -> String;
}

impl IntoSystemLabel<StrLabelMarker> for &str {
    fn into_label(self) -> String {
        self.to_string()
    }
}

impl IntoSystemLabel<StrLabelMarker> for String {
    fn into_label(self) -> String {
        self
    }
}

macro_rules! impl_into_system_label {
    ($($params:ident),*) => {
        impl<F, $($params: SystemParam),*> IntoSystemLabel<FnLabelMarker<($($params,)*)>> for F
        where
            for<'a, 'b> &'a mut F:
                FnMut($($params),*) +
                FnMut($(<$params as SystemParam>::Item<'b>),*)
        {
            fn into_label(self) -> String {
                std::any::type_name::<F>().to_string()
            }
        }
    }
}

impl_into_system_label!();
impl_into_system_label!(T1);
impl_into_system_label!(T1, T2);
impl_into_system_label!(T1, T2, T3);
impl_into_system_label!(T1, T2, T3, T4);
impl_into_system_label!(T1, T2, T3, T4, T5);
impl_into_system_label!(T1, T2, T3, T4, T5, T6);
impl_into_system_label!(T1, T2, T3, T4, T5, T6, T7);
impl_into_system_label!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_into_system_label!(T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_into_system_label!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);

// ─── Condition trait & FunctionCondition ─────────────────────────────────────

/// A DI-injected function that returns `bool`, used with `.run_if()`.
pub trait Condition {
    fn evaluate(&mut self, resources: &[RefCell<Box<dyn Any>>]) -> bool;
    fn prepare(&mut self, _index: &HashMap<TypeId, usize>) -> Vec<String> {
        Vec::new()
    }
    fn name(&self) -> &str {
        ""
    }
}

pub struct FunctionCondition<Input, F> {
    f: F,
    marker: PhantomData<fn() -> Input>,
    locals: HashMap<TypeId, Box<dyn Any>>,
    indices: Vec<usize>,
}

pub trait IntoCondition<Input> {
    type Condition: Condition;
    fn into_condition(self) -> Self::Condition;
}

impl_condition!();
impl_condition!(T1);
impl_condition!(T1, T2);
impl_condition!(T1, T2, T3);
impl_condition!(T1, T2, T3, T4);
impl_condition!(T1, T2, T3, T4, T5);

impl_into_condition!();
impl_into_condition!(T1);
impl_into_condition!(T1, T2);
impl_into_condition!(T1, T2, T3);
impl_into_condition!(T1, T2, T3, T4);
impl_into_condition!(T1, T2, T3, T4, T5);

// ─── ConditionalSystem ────────────────────────────────────────────────────────

/// Wraps a system with a run condition. The system only runs when the condition returns true.
pub struct ConditionalSystem<S: System, C: Condition> {
    system: S,
    condition: C,
}

impl<S: System, C: Condition> System for ConditionalSystem<S, C> {
    fn run(&mut self, resources: &[RefCell<Box<dyn Any>>]) {
        if self.condition.evaluate(resources) {
            self.system.run(resources);
        }
    }
    fn prepare(&mut self, index: &HashMap<TypeId, usize>) -> Vec<String> {
        let mut missing = self.condition.prepare(index);
        missing.extend(self.system.prepare(index));
        missing
    }
    fn name(&self) -> &str {
        self.system.name()
    }
    fn condition_name(&self) -> Option<&str> {
        let n = self.condition.name();
        if n.is_empty() { None } else { Some(n) }
    }
}

// ─── SystemDescriptor ─────────────────────────────────────────────────────────

/// Wraps a system with ordering metadata (label, before/after constraints).
pub struct SystemDescriptor<S: System + 'static> {
    pub system: S,
    pub label: Option<String>,
    pub befores: Vec<String>,
    pub afters: Vec<String>,
    pub requires: Vec<String>,
}

impl<S: System + 'static> SystemDescriptor<S> {
    pub fn label(mut self, lbl: impl Into<String>) -> Self {
        self.label = Some(lbl.into());
        self
    }
    pub fn before<M>(mut self, target: impl IntoSystemLabel<M>) -> Self {
        self.befores.push(target.into_label());
        self
    }
    pub fn after<M>(mut self, target: impl IntoSystemLabel<M>) -> Self {
        self.afters.push(target.into_label());
        self
    }
    pub fn requires_label<M>(mut self, target: impl IntoSystemLabel<M>) -> Self {
        self.requires.push(target.into_label());
        self
    }
    pub fn run_if<I2, C: Condition + 'static>(
        self,
        cond: impl IntoCondition<I2, Condition = C>,
    ) -> SystemDescriptor<ConditionalSystem<S, C>> {
        SystemDescriptor {
            system: ConditionalSystem {
                system: self.system,
                condition: cond.into_condition(),
            },
            label: self.label,
            befores: self.befores,
            afters: self.afters,
            requires: self.requires,
        }
    }
}

// ─── SystemExt — fluent API on IntoSystem ────────────────────────────────────

/// Extension trait giving any `IntoSystem` implementor the `.run_if()`, `.label()`,
/// `.before()`, and `.after()` fluent configuration methods.
pub trait SystemExt<I>: IntoSystem<I> + Sized
where
    Self::System: 'static,
{
    fn run_if<I2, C: Condition + 'static>(
        self,
        cond: impl IntoCondition<I2, Condition = C>,
    ) -> ConditionalSystem<Self::System, C> {
        ConditionalSystem {
            system: self.into_system(),
            condition: cond.into_condition(),
        }
    }

    fn label(self, lbl: impl Into<String>) -> SystemDescriptor<Self::System> {
        SystemDescriptor {
            system: self.into_system(),
            label: Some(lbl.into()),
            befores: vec![],
            afters: vec![],
            requires: vec![],
        }
    }

    fn before<M>(self, target: impl IntoSystemLabel<M>) -> SystemDescriptor<Self::System> {
        SystemDescriptor {
            system: self.into_system(),
            label: None,
            befores: vec![target.into_label()],
            afters: vec![],
            requires: vec![],
        }
    }

    fn after<M>(self, target: impl IntoSystemLabel<M>) -> SystemDescriptor<Self::System> {
        SystemDescriptor {
            system: self.into_system(),
            label: None,
            befores: vec![],
            afters: vec![target.into_label()],
            requires: vec![],
        }
    }

    fn requires_label<M>(self, target: impl IntoSystemLabel<M>) -> SystemDescriptor<Self::System> {
        SystemDescriptor {
            system: self.into_system(),
            label: None,
            befores: vec![],
            afters: vec![],
            requires: vec![target.into_label()],
        }
    }
}

impl<I, F: IntoSystem<I>> SystemExt<I> for F where F::System: 'static {}

// ─── IntoScheduledSystem — accepts fn / ConditionalSystem / SystemDescriptor ──

pub struct FnMarker<I>(PhantomData<I>);
pub struct CondMarker;
pub struct DescMarker;

/// Converts into a `StoredSystemEntry` (boxed system + optional ordering metadata).
pub trait IntoScheduledSystem<M> {
    fn into_stored(self) -> StoredSystemEntry;
}

/// Plain function / closure with no ordering metadata.
impl<I, F: IntoSystem<I>> IntoScheduledSystem<FnMarker<I>> for F
where
    F::System: 'static,
{
    fn into_stored(self) -> StoredSystemEntry {
        let sys = self.into_system();
        let name = sys.name().to_string();
        let condition_name = sys.condition_name().map(|s| s.to_string());
        StoredSystemEntry {
            system: Box::new(sys),
            name,
            label: None,
            befores: vec![],
            afters: vec![],
            requires: vec![],
            condition_name,
        }
    }
}

/// ConditionalSystem — no ordering metadata.
impl<S: System + 'static, C: Condition + 'static> IntoScheduledSystem<CondMarker>
    for ConditionalSystem<S, C>
{
    fn into_stored(self) -> StoredSystemEntry {
        let name = self.name().to_string();
        let condition_name = self.condition_name().map(|s| s.to_string());
        StoredSystemEntry {
            system: Box::new(self),
            name,
            label: None,
            befores: vec![],
            afters: vec![],
            requires: vec![],
            condition_name,
        }
    }
}

/// SystemDescriptor — carries label / before / after metadata.
impl<S: System + 'static> IntoScheduledSystem<DescMarker> for SystemDescriptor<S> {
    fn into_stored(self) -> StoredSystemEntry {
        let name = self.system.name().to_string();
        let condition_name = self.system.condition_name().map(|s| s.to_string());
        StoredSystemEntry {
            system: Box::new(self.system),
            name,
            label: self.label,
            befores: self.befores,
            afters: self.afters,
            requires: self.requires,
            condition_name,
        }
    }
}

// ─── StoredSystemEntry ────────────────────────────────────────────────────────

pub struct StoredSystemEntry {
    pub system: Box<dyn System>,
    pub name: String,
    pub label: Option<String>,
    pub befores: Vec<String>,
    pub afters: Vec<String>,
    pub requires: Vec<String>,
    pub condition_name: Option<String>,
}

// ─── Schedule sets ────────────────────────────────────────────────────────────

/// Execution phase within each timestep (run loop).
#[derive(Debug)]
pub enum ScheduleSet {
    Setup,
    PreInitialIntegration,
    InitialIntegration,
    PostInitialIntegration,
    PreExchange,
    Exchange,
    PreNeighbor,
    Neighbor,
    PreForce,
    Force,
    PostForce,
    PreFinalIntegration,
    FinalIntegration,
    PostFinalIntegration,
}

/// Execution phase during one-time setup (before the run loop).
#[derive(Debug)]
pub enum ScheduleSetupSet {
    PreSetup,
    Setup,
    PostSetup,
}

pub fn set_to_value(schedule_set: &ScheduleSet) -> u32 {
    match schedule_set {
        ScheduleSet::Setup => 0,
        ScheduleSet::PreInitialIntegration => 1,
        ScheduleSet::InitialIntegration => 2,
        ScheduleSet::PostInitialIntegration => 3,
        ScheduleSet::PreExchange => 4,
        ScheduleSet::Exchange => 5,
        ScheduleSet::PreNeighbor => 6,
        ScheduleSet::Neighbor => 7,
        ScheduleSet::PreForce => 8,
        ScheduleSet::Force => 9,
        ScheduleSet::PostForce => 10,
        ScheduleSet::PreFinalIntegration => 11,
        ScheduleSet::FinalIntegration => 12,
        ScheduleSet::PostFinalIntegration => 13,
    }
}

pub fn setup_set_to_value(schedule_set: &ScheduleSetupSet) -> u32 {
    match schedule_set {
        ScheduleSetupSet::PreSetup => 0,
        ScheduleSetupSet::Setup => 1,
        ScheduleSetupSet::PostSetup => 2,
    }
}

// ─── Simulation states ────────────────────────────────────────────────────────

/// The currently active simulation state.
pub struct CurrentState<S: Clone + PartialEq + 'static>(pub S);

/// The next state to transition to at the end of the step. Set via `NextState::set()`.
pub struct NextState<S: Clone + PartialEq + 'static>(pub Option<S>);

impl<S: Clone + PartialEq + 'static> NextState<S> {
    pub fn set(&mut self, state: S) {
        self.0 = Some(state);
    }
    pub fn clear(&mut self) {
        self.0 = None;
    }
}

/// Named condition struct for `in_state()`.
pub struct InStateCondition<S: Clone + PartialEq + 'static> {
    target: S,
    cond_name: String,
    index: usize,
}

pub struct InStateMarker;

impl<S: Clone + PartialEq + 'static> Condition for InStateCondition<S> {
    fn evaluate(&mut self, resources: &[RefCell<Box<dyn Any>>]) -> bool {
        let borrow = resources[self.index].borrow();
        let current = borrow.downcast_ref::<CurrentState<S>>().unwrap();
        current.0 == self.target
    }
    fn prepare(&mut self, index: &HashMap<TypeId, usize>) -> Vec<String> {
        let tid = TypeId::of::<CurrentState<S>>();
        match index.get(&tid) {
            Some(&idx) => { self.index = idx; vec![] }
            None => vec![std::any::type_name::<CurrentState<S>>().to_string()]
        }
    }
    fn name(&self) -> &str {
        &self.cond_name
    }
}

impl<S: Clone + PartialEq + 'static> IntoCondition<InStateMarker> for InStateCondition<S> {
    type Condition = Self;
    fn into_condition(self) -> Self { self }
}

/// Run condition: returns true when the current state equals `target`.
pub fn in_state<S: Clone + PartialEq + std::fmt::Debug + 'static>(
    target: S,
) -> InStateCondition<S> {
    let cond_name = format!("in_state({:?})", target);
    InStateCondition { target, cond_name, index: usize::MAX }
}

/// System that applies pending state transitions at end of step.
/// Register via `StatesPlugin<S>` or manually at `PostFinalIntegration`.
pub fn apply_state_transitions<S: Clone + PartialEq + 'static>(
    mut current: ResMut<CurrentState<S>>,
    mut next: ResMut<NextState<S>>,
) {
    if let Some(new_state) = next.0.take() {
        current.0 = new_state;
    }
}

/// Trait for enums that map 1:1 to named `[[run]]` stages.
///
/// Derive with `#[derive(StageEnum)]` and `#[stage("name")]` attributes.
pub trait StageName {
    /// Returns the stage name string for this variant.
    fn stage_name(&self) -> &'static str;
    /// Returns all stage names in variant order.
    fn stage_names() -> &'static [&'static str];
    /// Returns the number of stages.
    fn num_stages() -> usize;
}

/// Named condition struct for `in_stage()`.
pub struct InStageCondition {
    stage: String,
    cond_name: String,
    index: usize,
}

pub struct InStageMarker;

impl Condition for InStageCondition {
    fn evaluate(&mut self, resources: &[RefCell<Box<dyn Any>>]) -> bool {
        let borrow = resources[self.index].borrow();
        let sm = borrow.downcast_ref::<SchedulerManager>().unwrap();
        sm.stage_name.as_deref() == Some(self.stage.as_str())
    }
    fn prepare(&mut self, index: &HashMap<TypeId, usize>) -> Vec<String> {
        let tid = TypeId::of::<SchedulerManager>();
        match index.get(&tid) {
            Some(&idx) => { self.index = idx; vec![] }
            None => vec![std::any::type_name::<SchedulerManager>().to_string()]
        }
    }
    fn name(&self) -> &str {
        &self.cond_name
    }
}

impl IntoCondition<InStageMarker> for InStageCondition {
    type Condition = Self;
    fn into_condition(self) -> Self { self }
}

/// Run condition: returns true when the current stage name matches.
pub fn in_stage(name: &str) -> InStageCondition {
    let cond_name = format!("in_stage({})", name);
    InStageCondition { stage: name.to_string(), cond_name, index: usize::MAX }
}

/// Named condition struct for `first_stage_only()`.
pub struct FirstStageOnlyCondition {
    index: usize,
}

pub struct FirstStageOnlyMarker;

impl Condition for FirstStageOnlyCondition {
    fn evaluate(&mut self, resources: &[RefCell<Box<dyn Any>>]) -> bool {
        let borrow = resources[self.index].borrow();
        let sm = borrow.downcast_ref::<SchedulerManager>().unwrap();
        sm.index == 0
    }
    fn prepare(&mut self, index: &HashMap<TypeId, usize>) -> Vec<String> {
        let tid = TypeId::of::<SchedulerManager>();
        match index.get(&tid) {
            Some(&idx) => { self.index = idx; vec![] }
            None => vec![std::any::type_name::<SchedulerManager>().to_string()]
        }
    }
    fn name(&self) -> &str {
        "first_stage_only()"
    }
}

impl IntoCondition<FirstStageOnlyMarker> for FirstStageOnlyCondition {
    type Condition = Self;
    fn into_condition(self) -> Self { self }
}

/// Run condition: returns true only during the first stage (index == 0).
pub fn first_stage_only() -> FirstStageOnlyCondition {
    FirstStageOnlyCondition { index: usize::MAX }
}

/// System that detects state transitions and requests stage advancement.
///
/// Watches `CurrentState<S>` for changes; when a transition occurs, sets
/// `SchedulerManager::advance_requested = true`.
pub fn check_stage_advance<S: StageName + Clone + PartialEq + 'static>(
    current: Res<CurrentState<S>>,
    mut sm: ResMut<SchedulerManager>,
    mut prev: Local<Option<S>>,
) {
    if prev.as_ref() != Some(&current.0) {
        if prev.is_some() {
            // State changed — request advance to next stage
            sm.advance_requested = true;
        }
        *prev = Some(current.0.clone());
    }
}

// ─── Topological sort within a ScheduleSet group ─────────────────────────────

fn topo_sort_group(group: &mut Vec<(StoredSystemEntry, ScheduleSet)>) {
    let n = group.len();
    if n <= 1 {
        return;
    }

    let mut label_to_idx: HashMap<String, usize> = HashMap::new();
    for (i, (entry, _)) in group.iter().enumerate() {
        // Index by system name (enables function-handle-based ordering)
        label_to_idx.insert(entry.name.clone(), i);
        // Explicit labels override if present
        if let Some(lbl) = &entry.label {
            label_to_idx.insert(lbl.clone(), i);
        }
    }

    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for (i, (entry, _)) in group.iter().enumerate() {
        for b in &entry.befores {
            if let Some(&j) = label_to_idx.get(b) {
                adj[i].push(j);
                in_degree[j] += 1;
            }
        }
        for a in &entry.afters {
            if let Some(&j) = label_to_idx.get(a) {
                adj[j].push(i);
                in_degree[i] += 1;
            }
        }
    }

    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);

    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &nbr in &adj[node] {
            in_degree[nbr] -= 1;
            if in_degree[nbr] == 0 {
                queue.push_back(nbr);
            }
        }
    }

    if order.len() != n {
        panic!("Cycle detected in system ordering constraints within a ScheduleSet");
    }

    let mut temp: Vec<Option<(StoredSystemEntry, ScheduleSet)>> =
        group.drain(..).map(Some).collect();
    for idx in order {
        group.push(temp[idx].take().unwrap());
    }
}

// ─── Scheduler ────────────────────────────────────────────────────────────────

// ANCHOR: Scheduler
/// Manages system registration, resource storage, ordering, and per-step execution.
pub struct Scheduler {
    setup_systems: Vec<(StoredSystemEntry, ScheduleSetupSet)>,
    update_systems: Vec<(StoredSystemEntry, ScheduleSet)>,
    pub resources: Vec<RefCell<Box<dyn Any>>>,
    resource_index: HashMap<TypeId, usize>,
    print_schedule: bool,
    system_timings: Vec<f64>,
    timing_steps: usize,
    trace: bool,
    suppress_warnings: bool,
    stage_names: Vec<String>,
}
// ANCHOR_END: Scheduler

// Scheduler::default() is manually implemented because it appears in the book's ANCHOR blocks.
#[allow(clippy::derivable_impls)]
impl Default for Scheduler {
    fn default() -> Self {
        Scheduler {
            setup_systems: Vec::new(),
            update_systems: Vec::new(),
            resources: Vec::new(),
            resource_index: HashMap::new(),
            print_schedule: false,
            system_timings: Vec::new(),
            timing_steps: 0,
            trace: std::env::var("MDDEM_TRACE").is_ok(),
            suppress_warnings: std::env::var("MDDEM_SUPPRESS_WARNINGS").is_ok(),
            stage_names: Vec::new(),
        }
    }
}

// ANCHOR: SchedulerImpl
impl Scheduler {
    pub fn organize_systems(&mut self) {
        self.setup_systems
            .sort_by_key(|(_, f)| setup_set_to_value(f));
        self.update_systems.sort_by_key(|(_, f)| set_to_value(f));

        // Topo sort within each ScheduleSet group
        let all = std::mem::take(&mut self.update_systems);
        if all.is_empty() {
            // Still prepare setup systems
            let mut errors: Vec<String> = Vec::new();
            for (entry, _) in &mut self.setup_systems {
                for missing in entry.system.prepare(&self.resource_index) {
                    errors.push(format!("  System \"{}\" requires `{}`", entry.name, missing));
                }
            }
            if !errors.is_empty() {
                panic!("Schedule validation errors:\n{}", errors.join("\n"));
            }
            return;
        }

        let mut groups: Vec<Vec<(StoredSystemEntry, ScheduleSet)>> = Vec::new();
        for entry in all {
            let val = set_to_value(&entry.1);
            if let Some(last) = groups.last_mut() {
                if set_to_value(&last[0].1) == val {
                    last.push(entry);
                    continue;
                }
            }
            groups.push(vec![entry]);
        }

        let mut errors: Vec<String> = Vec::new();
        for mut group in groups {
            topo_sort_group(&mut group);

            // Validate requires_label: every required label/name must exist in this group
            let mut known_labels: Vec<&str> = Vec::new();
            for (entry, _) in &group {
                known_labels.push(&entry.name);
                if let Some(lbl) = &entry.label {
                    known_labels.push(lbl);
                }
            }
            for (entry, set) in &group {
                for req in &entry.requires {
                    if !known_labels.contains(&req.as_str()) {
                        errors.push(format!(
                            "  System \"{}\" in {:?} requires label \"{}\" which is not present in that ScheduleSet",
                            entry.name, set, req
                        ));
                    }
                }
            }

            self.update_systems.extend(group);
        }

        // Prepare all systems with cached resource indices, collecting missing resource errors
        for (entry, _) in &mut self.setup_systems {
            for missing in entry.system.prepare(&self.resource_index) {
                errors.push(format!("  System \"{}\" requires `{}`", entry.name, missing));
            }
        }
        for (entry, _) in &mut self.update_systems {
            for missing in entry.system.prepare(&self.resource_index) {
                errors.push(format!("  System \"{}\" requires `{}`", entry.name, missing));
            }
        }
        if !errors.is_empty() {
            panic!("Schedule validation errors:\n{}", errors.join("\n"));
        }

        // Initialize index-based timing vector
        self.system_timings = vec![0.0; self.update_systems.len()];

        // Emit non-blocking warnings for suspicious schedule configurations
        self.validate_schedule();
    }

    /// Returns warning strings for suspicious schedule configurations.
    /// Called at the end of `organize_systems()` to print warnings to stderr.
    fn schedule_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.update_systems.is_empty() {
            return warnings;
        }

        // Build a bool array: which ScheduleSets have at least one system?
        let mut has_systems = [false; 14];
        for (_, set) in &self.update_systems {
            has_systems[set_to_value(set) as usize] = true;
        }

        // Count total update systems (excluding Setup slot 0)
        let update_count: usize = has_systems[1..].iter().filter(|&&b| b).count();

        // 1. No Force systems — warn if schedule is non-empty but Force set is empty
        if update_count > 0 && !has_systems[set_to_value(&ScheduleSet::Force) as usize] {
            warnings.push(
                "[MDDEM Warning] No systems registered in the Force schedule set. \
                 Forces will not be computed. Did you forget a force plugin?"
                    .to_string(),
            );
        }

        // 2. Asymmetric Verlet — InitialIntegration without FinalIntegration or vice versa
        let has_initial = has_systems[set_to_value(&ScheduleSet::InitialIntegration) as usize]
            || has_systems[set_to_value(&ScheduleSet::PreInitialIntegration) as usize];
        let has_final = has_systems[set_to_value(&ScheduleSet::FinalIntegration) as usize]
            || has_systems[set_to_value(&ScheduleSet::PostFinalIntegration) as usize];
        if has_initial && !has_final {
            warnings.push(
                "[MDDEM Warning] InitialIntegration has systems but FinalIntegration is empty. \
                 This produces an asymmetric Verlet integration."
                    .to_string(),
            );
        } else if !has_initial && has_final {
            warnings.push(
                "[MDDEM Warning] FinalIntegration has systems but InitialIntegration is empty. \
                 This produces an asymmetric Verlet integration."
                    .to_string(),
            );
        }

        // 3. No integrator — many update systems but no integration at all
        if update_count > 2 && !has_initial && !has_final {
            warnings.push(
                "[MDDEM Warning] Schedule has update systems but no integrator \
                 (neither InitialIntegration nor FinalIntegration has systems). \
                 Particles will not move."
                    .to_string(),
            );
        }

        warnings
    }

    fn validate_schedule(&self) {
        if self.suppress_warnings {
            return;
        }
        for warning in self.schedule_warnings() {
            eprintln!("{}", warning);
        }
    }

    pub fn setup(&mut self) {
        for (entry, _set) in self.setup_systems.iter_mut() {
            entry.system.run(&self.resources);
        }
    }

    pub fn run(&mut self) {
        for (idx, (entry, set)) in self.update_systems.iter_mut().enumerate() {
            if self.trace {
                eprintln!("[step {}] {:?}: {}", self.timing_steps, set, entry.name);
            }
            let t0 = std::time::Instant::now();
            entry.system.run(&self.resources);
            self.system_timings[idx] += t0.elapsed().as_secs_f64();
        }
        self.timing_steps += 1;
    }

    pub fn start(&mut self) {
        self.add_scheduler_manager();
        let mut schedule_state = SchedulerState::Setup;
        while !matches!(schedule_state, SchedulerState::End) {
            if matches!(schedule_state, SchedulerState::Setup) {
                self.organize_systems();
                if self.print_schedule {
                    self.write_dot("schedule.dot");
                }
                self.setup();

                let sm_idx = self.resource_index[&TypeId::of::<SchedulerManager>()];
                let mut binding = self.resources[sm_idx].borrow_mut();
                let sm = binding.downcast_mut::<SchedulerManager>().unwrap();
                sm.state = SchedulerState::Run;
            }

            if matches!(schedule_state, SchedulerState::Run) {
                self.run();
            }

            let sm_idx = self.resource_index[&TypeId::of::<SchedulerManager>()];
            let mut binding = self.resources[sm_idx].borrow_mut();
            let sm = binding.downcast_mut::<SchedulerManager>().unwrap();
            schedule_state = sm.state;
        }

        // Print per-system timing breakdown
        if self.timing_steps > 0 && !self.system_timings.is_empty() {
            let total: f64 = self.system_timings.iter().sum();
            let mut sorted: Vec<_> = self.update_systems.iter()
                .zip(self.system_timings.iter())
                .map(|((entry, _), &time)| (&entry.name, time))
                .collect::<Vec<_>>();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!("\n--- Per-system timing ({} steps) ---", self.timing_steps);
            println!("{:<50} {:>10} {:>8}", "System", "Time(s)", "%");
            println!("{}", "-".repeat(70));
            for (name, time) in &sorted {
                let pct = *time / total * 100.0;
                println!("{:<50} {:>10.4} {:>7.1}%", name, time, pct);
            }
            println!("{}", "-".repeat(70));
            println!("{:<50} {:>10.4} {:>7.1}%", "TOTAL", total, 100.0);
        }
    }

    pub fn add_setup_system<M>(
        &mut self,
        system: impl IntoScheduledSystem<M>,
        schedule_set: ScheduleSetupSet,
    ) {
        self.setup_systems
            .push((system.into_stored(), schedule_set));
    }

    pub fn add_update_system<M>(
        &mut self,
        system: impl IntoScheduledSystem<M>,
        schedule_set: ScheduleSet,
    ) {
        self.update_systems
            .push((system.into_stored(), schedule_set));
    }

    pub fn remove_update_system<I, S: System + 'static>(
        &mut self,
        system: impl IntoSystem<I, System = S>,
    ) {
        let sys = system.into_system();
        let name = sys.name();
        self.update_systems
            .retain(|(entry, _)| entry.name != name);
    }

    pub fn remove_update_system_by_label(&mut self, label: &str) {
        self.update_systems
            .retain(|(entry, _)| entry.label.as_deref() != Some(label));
    }

    pub fn add_scheduler_manager(&mut self) {
        self.add_resource(SchedulerManager::new());
    }

    pub fn add_resource<R: 'static>(&mut self, res: R) {
        let type_id = TypeId::of::<R>();
        if let Some(&idx) = self.resource_index.get(&type_id) {
            self.resources[idx] = RefCell::new(Box::new(res));
        } else {
            let idx = self.resources.len();
            self.resources.push(RefCell::new(Box::new(res)));
            self.resource_index.insert(type_id, idx);
        }
    }

    pub fn get_mut_resource(&mut self, res: TypeId) -> Option<&RefCell<Box<dyn Any>>> {
        self.resource_index
            .get(&res)
            .map(|&idx| &self.resources[idx])
    }

    pub fn get_resource_ref<R: 'static>(&self) -> Option<std::cell::Ref<'_, R>> {
        self.resource_index
            .get(&TypeId::of::<R>())
            .map(|&idx| {
                std::cell::Ref::map(self.resources[idx].borrow(), |b| {
                    b.downcast_ref::<R>().unwrap()
                })
            })
    }

    pub fn enable_schedule_print(&mut self) {
        self.print_schedule = true;
    }

    pub fn set_stage_names(&mut self, names: &[&str]) {
        self.stage_names = names.iter().map(|s| s.to_string()).collect();
    }

    fn short_name(full: &str) -> String {
        let parts: Vec<&str> = full.rsplitn(3, "::").collect();
        match parts.len() {
            0 => full.to_string(),
            1 => parts[0].to_string(),
            _ => format!("{}::{}", parts[1], parts[0]),
        }
    }

    fn condition_short_name(full: &str) -> String {
        // in_state closure: "mddem_scheduler::in_state::{{closure}}" → "in_state(..)"
        if full.contains("in_state") && full.contains("{{closure}}") {
            return "in_state(..)".to_string();
        }
        // first_stage_only closure: "mddem_scheduler::first_stage_only::{{closure}}" → "first_stage_only()"
        if full.contains("first_stage_only") && full.contains("{{closure}}") {
            return "first_stage_only()".to_string();
        }
        // in_stage closure
        if full.contains("in_stage") && full.contains("{{closure}}") {
            return "in_stage(..)".to_string();
        }
        // Generic closure fallback
        if full.contains("{{closure}}") {
            // Try to extract the function name before ::{{closure}}
            if let Some(pos) = full.rfind("::{{closure}}") {
                let prefix = &full[..pos];
                if let Some(last_sep) = prefix.rfind("::") {
                    return format!("{}(..)", &prefix[last_sep + 2..]);
                }
                return format!("{}(..)", prefix);
            }
        }
        Self::short_name(full)
    }

    pub fn print_schedule(&self) {
        println!("\n═══ Setup Systems ═══");
        let mut last_set: Option<u32> = None;
        for (entry, set) in &self.setup_systems {
            let val = setup_set_to_value(set);
            if last_set != Some(val) {
                println!("  [{:?}]", set);
                last_set = Some(val);
            }
            let short = Self::short_name(&entry.name);
            if let Some(lbl) = &entry.label {
                print!("    {}  [{}]", short, lbl);
            } else {
                print!("    {}", short);
            }
            if !entry.afters.is_empty() {
                print!("  after: {}", entry.afters.join(", "));
            }
            if !entry.befores.is_empty() {
                print!("  before: {}", entry.befores.join(", "));
            }
            if let Some(cond) = &entry.condition_name {
                print!("  run_if: {}", Self::condition_short_name(cond));
            }
            println!();
        }

        println!("\n═══ Update Systems (per-step) ═══");
        let mut last_set: Option<u32> = None;
        for (entry, set) in &self.update_systems {
            let val = set_to_value(set);
            if last_set != Some(val) {
                println!("  [{:?}]", set);
                last_set = Some(val);
            }
            let short = Self::short_name(&entry.name);
            if let Some(lbl) = &entry.label {
                print!("    {}  [{}]", short, lbl);
            } else {
                print!("    {}", short);
            }
            if !entry.afters.is_empty() {
                print!("  after: {}", entry.afters.join(", "));
            }
            if !entry.befores.is_empty() {
                print!("  before: {}", entry.befores.join(", "));
            }
            if let Some(cond) = &entry.condition_name {
                print!("  run_if: {}", Self::condition_short_name(cond));
            }
            println!();
        }
        println!();
    }

    /// Returns the stage index a condition belongs to, or `None` (= show in all stages).
    fn condition_stage_index(&self, cond_name: &str) -> Option<usize> {
        Self::condition_stage_index_static(cond_name, &self.stage_names)
    }

    fn condition_stage_index_static(cond_name: &str, stage_names: &[String]) -> Option<usize> {
        if stage_names.is_empty() {
            return None;
        }
        // "in_state(Insert)" → match variant name to stage name (case-insensitive)
        if let Some(variant) = cond_name.strip_prefix("in_state(").and_then(|s| s.strip_suffix(')')) {
            let variant_lower = variant.to_lowercase();
            return stage_names.iter().position(|s| s.to_lowercase() == variant_lower);
        }
        // "in_stage(relax)" → exact match
        if let Some(name) = cond_name.strip_prefix("in_stage(").and_then(|s| s.strip_suffix(')')) {
            return stage_names.iter().position(|s| s == name);
        }
        // "first_stage_only()" → first stage
        if cond_name == "first_stage_only()" {
            return Some(0);
        }
        None
    }

    /// Returns true if an update system should appear in the given stage.
    fn system_visible_in_stage(&self, entry: &StoredSystemEntry, stage_idx: usize) -> bool {
        match &entry.condition_name {
            None => true, // unconditional → visible in all stages
            Some(cond) => {
                match self.condition_stage_index(cond) {
                    Some(idx) => idx == stage_idx, // stage-specific → only in matching stage
                    None => true, // unknown condition → show in all stages
                }
            }
        }
    }

    /// Returns true if a setup system should appear in the given stage.
    /// Setup runs every stage, so unconditional setup systems are visible in all stages.
    fn setup_visible_in_stage(&self, entry: &StoredSystemEntry, stage_idx: usize) -> bool {
        self.system_visible_in_stage(entry, stage_idx)
    }

    pub fn write_dot(&self, path: &str) {
        use std::io::Write;
        let mut out = String::new();
        out.push_str("digraph schedule {\n");
        out.push_str("    node [shape=box, style=filled, fillcolor=lightyellow];\n\n");

        // Helper to make valid DOT node IDs
        let node_id = |prefix: &str, idx: usize| -> String { format!("{}_{}", prefix, idx) };

        // Capture stage_names for use in closures
        let stage_names = &self.stage_names;

        // Check if a condition matches a specific stage index
        let cond_matches_stage = |cond: &str, si: usize| -> bool {
            Self::condition_stage_index_static(cond, stage_names) == Some(si)
        };

        // Helper to build a node label with optional condition annotation.
        // When `current_stage` is Some, conditions that match that stage are suppressed
        // (e.g. first_stage_only() in stage 0 is redundant).
        let make_label = |entry: &StoredSystemEntry, current_stage: Option<usize>| -> String {
            let short = Self::short_name(&entry.name);
            let mut label = if let Some(lbl) = &entry.label {
                format!("{}\\n[{}]", short, lbl)
            } else {
                short.to_string()
            };
            if let Some(cond) = &entry.condition_name {
                let show_cond = match current_stage {
                    Some(si) => !cond_matches_stage(cond, si),
                    None => true,
                };
                if show_cond {
                    label.push_str(&format!("\\nrun_if: {}", Self::condition_short_name(cond)));
                }
            }
            label
        };

        // Helper to build node style (dashed border for conditional systems).
        // Suppresses dashed border when the condition matches the current stage.
        let node_style = |entry: &StoredSystemEntry, current_stage: Option<usize>| -> String {
            let is_conditional = match &entry.condition_name {
                Some(cond) => match current_stage {
                    Some(si) => !cond_matches_stage(cond, si),
                    None => true,
                },
                None => false,
            };
            if is_conditional {
                "style=\"filled,dashed\"".to_string()
            } else {
                "style=filled".to_string()
            }
        };

        // Build setup groups once
        let mut setup_groups: Vec<(String, Vec<(usize, &StoredSystemEntry)>)> = Vec::new();
        for (i, (entry, set)) in self.setup_systems.iter().enumerate() {
            let set_name = format!("{:?}", set);
            if let Some(last) = setup_groups.last_mut() {
                if last.0 == set_name {
                    last.1.push((i, entry));
                    continue;
                }
            }
            setup_groups.push((set_name, vec![(i, entry)]));
        }

        if self.stage_names.is_empty() {
            // No stages — single-loop layout with TB
            out.insert_str("digraph schedule {\n".len(), "    rankdir=TB;\n");
            self.write_dot_flat(&mut out, &node_id, &make_label, &node_style, &setup_groups);
        } else {
            // Per-stage layout: stages left-to-right, pipelines top-to-bottom
            out.insert_str("digraph schedule {\n".len(), "    rankdir=TB;\n    newrank=true;\n");
            self.write_dot_per_stage(&mut out, &node_id, &make_label, &node_style, &setup_groups);
        }

        // Legend
        out.push_str("    subgraph cluster_legend {\n");
        out.push_str("        label=\"Legend\";\n");
        out.push_str("        style=filled; fillcolor=white;\n");
        out.push_str("        node [shape=plaintext, style=\"\", fillcolor=white];\n");
        out.push_str("        legend_1 [label=\"Blue bold = execution order\"];\n");
        out.push_str("        legend_2 [label=\"Red dashed = before/after constraint\"];\n");
        out.push_str("        legend_3 [label=\"Purple dotted = requires constraint\"];\n");
        out.push_str("        legend_4 [label=\"Green bold = run loop\"];\n");
        out.push_str("        legend_5 [label=\"Dashed border = conditional (run_if)\"];\n");
        out.push_str("        legend_1 -> legend_2 -> legend_3 -> legend_4 -> legend_5 [style=invis];\n");
        out.push_str("    }\n\n");

        out.push_str("}\n");

        let mut file = std::fs::File::create(path).expect("Failed to create DOT file");
        file.write_all(out.as_bytes())
            .expect("Failed to write DOT file");
        println!("Schedule DOT file written to: {}", path);
    }

    /// Flat single-loop DOT layout (no stages). Setup shown once, then one run loop.
    fn write_dot_flat(
        &self,
        out: &mut String,
        node_id: &dyn Fn(&str, usize) -> String,
        make_label: &dyn Fn(&StoredSystemEntry, Option<usize>) -> String,
        node_style: &dyn Fn(&StoredSystemEntry, Option<usize>) -> String,
        setup_groups: &[(String, Vec<(usize, &StoredSystemEntry)>)],
    ) {
        // Setup systems
        for (set_name, entries) in setup_groups {
            out.push_str(&format!("    subgraph cluster_setup_{} {{\n", set_name));
            out.push_str(&format!("        label=\"Setup: {}\";\n", set_name));
            out.push_str("        style=filled; fillcolor=lightblue;\n");
            for &(i, entry) in entries {
                out.push_str(&format!(
                    "        {} [label=\"{}\", {}];\n",
                    node_id("setup", i), make_label(entry, None), node_style(entry, None)
                ));
            }
            out.push_str("    }\n\n");
        }
        for i in 0..setup_groups.len().saturating_sub(1) {
            let tail = node_id("setup", setup_groups[i].1.last().unwrap().0);
            let head = node_id("setup", setup_groups[i + 1].1.first().unwrap().0);
            out.push_str(&format!("    {} -> {} [color=blue, style=bold];\n", tail, head));
        }
        for (_set_name, entries) in setup_groups {
            for w in entries.windows(2) {
                out.push_str(&format!(
                    "    {} -> {} [color=blue, style=bold];\n",
                    node_id("setup", w[0].0), node_id("setup", w[1].0)
                ));
            }
        }

        // Update systems
        let mut update_groups: Vec<(String, Vec<(usize, &StoredSystemEntry)>)> = Vec::new();
        for (i, (entry, set)) in self.update_systems.iter().enumerate() {
            let set_name = format!("{:?}", set);
            if let Some(last) = update_groups.last_mut() {
                if last.0 == set_name {
                    last.1.push((i, entry));
                    continue;
                }
            }
            update_groups.push((set_name, vec![(i, entry)]));
        }

        for (set_name, entries) in &update_groups {
            out.push_str(&format!("    subgraph cluster_{} {{\n", set_name));
            out.push_str(&format!("        label=\"{}\";\n", set_name));
            out.push_str("        style=filled; fillcolor=lightyellow;\n");
            for &(i, entry) in entries {
                out.push_str(&format!(
                    "        {} [label=\"{}\", {}];\n",
                    node_id("update", i), make_label(entry, None), node_style(entry, None)
                ));
            }
            out.push_str("    }\n\n");
        }

        for (_set_name, entries) in &update_groups {
            for w in entries.windows(2) {
                out.push_str(&format!(
                    "    {} -> {} [color=blue, style=bold];\n",
                    node_id("update", w[0].0), node_id("update", w[1].0)
                ));
            }
        }

        // Before/after constraint edges
        let mut label_to_node: HashMap<String, String> = HashMap::new();
        for (i, (entry, _)) in self.update_systems.iter().enumerate() {
            if let Some(lbl) = &entry.label {
                label_to_node.insert(lbl.clone(), node_id("update", i));
            }
        }
        for (i, (entry, _)) in self.update_systems.iter().enumerate() {
            let from = node_id("update", i);
            for b in &entry.befores {
                if let Some(target) = label_to_node.get(b) {
                    out.push_str(&format!(
                        "    {} -> {} [color=red, style=dashed, label=\"before\"];\n", from, target
                    ));
                }
            }
            for a in &entry.afters {
                if let Some(source) = label_to_node.get(a) {
                    out.push_str(&format!(
                        "    {} -> {} [color=red, style=dashed, label=\"after\"];\n", source, from
                    ));
                }
            }
        }
        for (i, (entry, _)) in self.update_systems.iter().enumerate() {
            let from = node_id("update", i);
            for req in &entry.requires {
                if let Some(target) = label_to_node.get(req) {
                    out.push_str(&format!(
                        "    {} -> {} [color=purple, style=dotted, label=\"requires\", constraint=false];\n",
                        from, target
                    ));
                }
            }
        }

        // Cluster-to-cluster edges
        let tails: Vec<String> = update_groups.iter()
            .map(|(_, e)| node_id("update", e.last().unwrap().0)).collect();
        let heads: Vec<String> = update_groups.iter()
            .map(|(_, e)| node_id("update", e.first().unwrap().0)).collect();
        for i in 0..tails.len().saturating_sub(1) {
            out.push_str(&format!(
                "    {} -> {} [color=blue, style=bold];\n", tails[i], heads[i + 1]
            ));
        }
        if tails.len() >= 2 {
            out.push_str(&format!(
                "    {} -> {} [color=green, style=bold, label=\"run loop\", constraint=false];\n",
                tails.last().unwrap(), heads.first().unwrap()
            ));
        }
        if !setup_groups.is_empty() && !heads.is_empty() {
            let last_setup = setup_groups.last().unwrap();
            let last_setup_node = node_id("setup", last_setup.1.last().unwrap().0);
            out.push_str(&format!(
                "    {} -> {} [color=blue, style=bold, label=\"start run\"];\n",
                last_setup_node, heads[0]
            ));
        }
    }

    /// Per-stage DOT layout: stages left-to-right, each with Setup → Run pipeline top-to-bottom.
    fn write_dot_per_stage(
        &self,
        out: &mut String,
        node_id: &dyn Fn(&str, usize) -> String,
        make_label: &dyn Fn(&StoredSystemEntry, Option<usize>) -> String,
        node_style: &dyn Fn(&StoredSystemEntry, Option<usize>) -> String,
        _setup_groups: &[(String, Vec<(usize, &StoredSystemEntry)>)],
    ) {
        let stage_colors = ["#E8F5E9", "#E3F2FD", "#FFF3E0", "#F3E5F5", "#FFFDE7", "#FCE4EC"];

        // Collect first-node IDs from each stage for rank=same alignment
        let mut stage_first_nodes: Vec<String> = Vec::new();

        for (si, stage_name) in self.stage_names.iter().enumerate() {
            let prefix = format!("s{}", si);
            let fill = stage_colors[si % stage_colors.len()];

            out.push_str(&format!("    subgraph cluster_stage_{} {{\n", si));
            out.push_str(&format!("        label=\"Stage: {}\";\n", stage_name));
            out.push_str(&format!("        style=filled; fillcolor=\"{}\";\n", fill));
            out.push_str("        color=darkgreen; penwidth=2;\n");

            // ── Setup systems within this stage ──────────────────────────
            let mut stage_setup_groups: Vec<(String, Vec<(usize, &StoredSystemEntry)>)> = Vec::new();
            for (i, (entry, set)) in self.setup_systems.iter().enumerate() {
                if !self.setup_visible_in_stage(entry, si) {
                    continue;
                }
                let set_name = format!("{:?}", set);
                if let Some(last) = stage_setup_groups.last_mut() {
                    if last.0 == set_name {
                        last.1.push((i, entry));
                        continue;
                    }
                }
                stage_setup_groups.push((set_name, vec![(i, entry)]));
            }

            for (set_name, entries) in &stage_setup_groups {
                let setup_prefix = format!("{}setup", prefix);
                out.push_str(&format!("        subgraph cluster_{}_setup_{} {{\n", set_name, si));
                out.push_str(&format!("            label=\"Setup: {}\";\n", set_name));
                out.push_str("            style=filled; fillcolor=lightblue;\n");
                for &(i, entry) in entries {
                    out.push_str(&format!(
                        "            {} [label=\"{}\", {}];\n",
                        node_id(&setup_prefix, i), make_label(entry, Some(si)), node_style(entry, Some(si))
                    ));
                }
                out.push_str("        }\n");
            }

            // ── Update systems within this stage ─────────────────────────
            let mut stage_groups: Vec<(String, Vec<(usize, &StoredSystemEntry)>)> = Vec::new();
            for (i, (entry, set)) in self.update_systems.iter().enumerate() {
                if !self.system_visible_in_stage(entry, si) {
                    continue;
                }
                let set_name = format!("{:?}", set);
                if let Some(last) = stage_groups.last_mut() {
                    if last.0 == set_name {
                        last.1.push((i, entry));
                        continue;
                    }
                }
                stage_groups.push((set_name, vec![(i, entry)]));
            }

            for (set_name, entries) in &stage_groups {
                out.push_str(&format!("        subgraph cluster_{}_s{} {{\n", set_name, si));
                out.push_str(&format!("            label=\"{}\";\n", set_name));
                out.push_str("            style=filled; fillcolor=lightyellow;\n");
                for &(i, entry) in entries {
                    out.push_str(&format!(
                        "            {} [label=\"{}\", {}];\n",
                        node_id(&prefix, i), make_label(entry, Some(si)), node_style(entry, Some(si))
                    ));
                }
                out.push_str("        }\n");
            }

            out.push_str("    }\n\n");

            // Track first node for rank=same alignment
            let setup_prefix = format!("{}setup", prefix);
            let first_node_id = if let Some(first_sg) = stage_setup_groups.first() {
                node_id(&setup_prefix, first_sg.1.first().unwrap().0)
            } else if let Some(first_ug) = stage_groups.first() {
                node_id(&prefix, first_ug.1.first().unwrap().0)
            } else {
                continue; // empty stage
            };
            stage_first_nodes.push(first_node_id.clone());

            // ── Intra-stage edges ────────────────────────────────────────

            // Setup intra-group edges
            for (_set_name, entries) in &stage_setup_groups {
                for w in entries.windows(2) {
                    out.push_str(&format!(
                        "    {} -> {} [color=blue, style=bold];\n",
                        node_id(&setup_prefix, w[0].0), node_id(&setup_prefix, w[1].0)
                    ));
                }
            }
            // Setup inter-group edges
            for g in 0..stage_setup_groups.len().saturating_sub(1) {
                let tail = node_id(&setup_prefix, stage_setup_groups[g].1.last().unwrap().0);
                let head = node_id(&setup_prefix, stage_setup_groups[g + 1].1.first().unwrap().0);
                out.push_str(&format!("    {} -> {} [color=blue, style=bold];\n", tail, head));
            }

            // Setup → Run transition within this stage
            if let (Some(last_sg), Some(first_ug)) = (stage_setup_groups.last(), stage_groups.first()) {
                let tail = node_id(&setup_prefix, last_sg.1.last().unwrap().0);
                let head = node_id(&prefix, first_ug.1.first().unwrap().0);
                out.push_str(&format!(
                    "    {} -> {} [color=blue, style=bold, label=\"start run\"];\n", tail, head
                ));
            }

            // Update intra-group edges
            for (_set_name, entries) in &stage_groups {
                for w in entries.windows(2) {
                    out.push_str(&format!(
                        "    {} -> {} [color=blue, style=bold];\n",
                        node_id(&prefix, w[0].0), node_id(&prefix, w[1].0)
                    ));
                }
            }
            // Update inter-group edges
            for g in 0..stage_groups.len().saturating_sub(1) {
                let tail = node_id(&prefix, stage_groups[g].1.last().unwrap().0);
                let head = node_id(&prefix, stage_groups[g + 1].1.first().unwrap().0);
                out.push_str(&format!("    {} -> {} [color=blue, style=bold];\n", tail, head));
            }

            // Before/after constraint edges
            let mut label_to_node: HashMap<String, String> = HashMap::new();
            for (i, (entry, _)) in self.update_systems.iter().enumerate() {
                if !self.system_visible_in_stage(entry, si) { continue; }
                if let Some(lbl) = &entry.label {
                    label_to_node.insert(lbl.clone(), node_id(&prefix, i));
                }
            }
            for (i, (entry, _)) in self.update_systems.iter().enumerate() {
                if !self.system_visible_in_stage(entry, si) { continue; }
                let from = node_id(&prefix, i);
                for b in &entry.befores {
                    if let Some(target) = label_to_node.get(b) {
                        out.push_str(&format!(
                            "    {} -> {} [color=red, style=dashed, label=\"before\"];\n", from, target
                        ));
                    }
                }
                for a in &entry.afters {
                    if let Some(source) = label_to_node.get(a) {
                        out.push_str(&format!(
                            "    {} -> {} [color=red, style=dashed, label=\"after\"];\n", source, from
                        ));
                    }
                }
            }
            for (i, (entry, _)) in self.update_systems.iter().enumerate() {
                if !self.system_visible_in_stage(entry, si) { continue; }
                let from = node_id(&prefix, i);
                for req in &entry.requires {
                    if let Some(target) = label_to_node.get(req) {
                        out.push_str(&format!(
                            "    {} -> {} [color=purple, style=dotted, label=\"requires\", constraint=false];\n",
                            from, target
                        ));
                    }
                }
            }

            // Run loop: last update → first update
            if let (Some(first_ug), Some(last_ug)) = (stage_groups.first(), stage_groups.last()) {
                let first_run = node_id(&prefix, first_ug.1.first().unwrap().0);
                let last_run = node_id(&prefix, last_ug.1.last().unwrap().0);

                if stage_groups.len() >= 2 || first_ug.1.len() >= 2 {
                    out.push_str(&format!(
                        "    {} -> {} [color=green, style=bold, label=\"run loop\", constraint=false];\n",
                        last_run, first_run
                    ));
                }
            }
        }

        // Inter-stage "next stage" edges (constraint=false so they go sideways)
        // We connect from the last run node of stage N to the first setup node of stage N+1.
        for si in 0..self.stage_names.len().saturating_sub(1) {
            let this_prefix = format!("s{}", si);
            let next_prefix = format!("s{}", si + 1);
            let next_setup_prefix = format!("{}setup", next_prefix);

            // Find last update node of current stage
            let last_node = self.update_systems.iter().enumerate().rev()
                .find(|(_, (entry, _))| self.system_visible_in_stage(entry, si))
                .map(|(i, _)| node_id(&this_prefix, i));

            // Find first node of next stage (setup first, then update)
            let first_node = self.setup_systems.iter().enumerate()
                .find(|(_, (entry, _))| self.setup_visible_in_stage(entry, si + 1))
                .map(|(i, _)| node_id(&next_setup_prefix, i))
                .or_else(|| self.update_systems.iter().enumerate()
                    .find(|(_, (entry, _))| self.system_visible_in_stage(entry, si + 1))
                    .map(|(i, _)| node_id(&next_prefix, i)));

            if let (Some(from), Some(to)) = (last_node, first_node) {
                out.push_str(&format!(
                    "    {} -> {} [color=darkgreen, style=bold, penwidth=3, label=\"next stage\", constraint=false];\n",
                    from, to
                ));
            }
        }

        // Force stages side-by-side using rank=same on first nodes
        if stage_first_nodes.len() >= 2 {
            out.push_str(&format!(
                "    {{ rank=same; {} }}\n\n",
                stage_first_nodes.join("; ")
            ));
        }
    }
}
// ANCHOR_END: SchedulerImpl
// ANCHOR_END: All

// ─── SchedulerState / SchedulerManager ───────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum SchedulerState {
    Setup,
    Run,
    End,
}

/// Tracks the current run stage index and scheduler state (Setup/Run/End).
pub struct SchedulerManager {
    pub state: SchedulerState,
    pub index: usize,
    /// Name of the current stage from `[[run]]` config (if set).
    pub stage_name: Option<String>,
    /// When true, the scheduler will advance to the next stage at end of step.
    pub advance_requested: bool,
}

impl Default for SchedulerManager {
    fn default() -> Self {
        SchedulerManager {
            state: SchedulerState::Setup,
            index: 0,
            stage_name: None,
            advance_requested: false,
        }
    }
}

impl SchedulerManager {
    pub fn new() -> Self {
        Self::default()
    }
}

// ─── Prelude ──────────────────────────────────────────────────────────────────

pub mod prelude {
    pub use crate::{
        apply_state_transitions,
        check_stage_advance,
        first_stage_only,
        in_stage,
        in_state,
        ConditionalSystem,
        // Simulation states
        CurrentState,
        IntoScheduledSystem,
        // System label (function-handle or string ordering)
        IntoSystemLabel,
        Local,
        NextState,
        Res,
        ResMut,
        // System scheduling
        ScheduleSet,
        ScheduleSetupSet,
        // Core DI
        Scheduler,
        SchedulerManager,
        SchedulerState,
        // Stage enum trait
        StageName,
        // System ordering
        SystemDescriptor,
        // Run conditions
        SystemExt,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MyResource(i32);

    fn system_requiring_resource(_res: Res<MyResource>) {}

    #[test]
    #[should_panic(expected = "Schedule validation errors")]
    fn missing_resource_panics_at_organize() {
        let mut scheduler = Scheduler::default();
        scheduler.add_update_system(system_requiring_resource, ScheduleSet::Force);
        scheduler.organize_systems();
    }

    fn system_with_optional_resource(res: Option<Res<MyResource>>) {
        assert!(res.is_none());
    }

    #[test]
    fn optional_resource_works_when_missing() {
        let mut scheduler = Scheduler::default();
        scheduler.add_update_system(system_with_optional_resource, ScheduleSet::Force);
        scheduler.organize_systems();
        scheduler.add_scheduler_manager();
        scheduler.organize_systems();
        scheduler.run();
    }

    fn system_with_optional_present(res: Option<Res<MyResource>>) {
        assert!(res.is_some());
        assert_eq!(res.unwrap().0, 42);
    }

    #[test]
    fn optional_resource_works_when_present() {
        let mut scheduler = Scheduler::default();
        scheduler.add_resource(MyResource(42));
        scheduler.add_update_system(system_with_optional_present, ScheduleSet::Force);
        scheduler.add_scheduler_manager();
        scheduler.organize_systems();
        scheduler.run();
    }

    #[test]
    fn remove_update_system_by_label() {
        let mut scheduler = Scheduler::default();
        scheduler.add_update_system(
            system_requiring_resource.label("my_system"),
            ScheduleSet::Force,
        );
        assert_eq!(scheduler.update_systems.len(), 1);
        scheduler.remove_update_system_by_label("my_system");
        assert_eq!(scheduler.update_systems.len(), 0);
    }

    fn dummy_system(_res: Res<MyResource>) {}

    #[test]
    fn remove_update_system_by_function() {
        let mut scheduler = Scheduler::default();
        scheduler.add_update_system(dummy_system, ScheduleSet::Force);
        assert_eq!(scheduler.update_systems.len(), 1);
        scheduler.remove_update_system(dummy_system);
        assert_eq!(scheduler.update_systems.len(), 0);
    }

    // ─── requires_label tests ────────────────────────────────────────────────

    fn force_a() {}
    fn force_b() {}

    #[test]
    #[should_panic(expected = "requires label")]
    fn requires_label_panics_when_missing() {
        let mut scheduler = Scheduler::default();
        scheduler.suppress_warnings = true;
        scheduler.add_update_system(
            force_b.label("force_b").requires_label("force_a"),
            ScheduleSet::Force,
        );
        scheduler.organize_systems();
    }

    #[test]
    fn requires_label_passes_when_present() {
        let mut scheduler = Scheduler::default();
        scheduler.suppress_warnings = true;
        scheduler.add_update_system(force_a.label("force_a"), ScheduleSet::Force);
        scheduler.add_update_system(
            force_b.label("force_b").requires_label("force_a"),
            ScheduleSet::Force,
        );
        scheduler.organize_systems();
    }

    fn always_true() -> bool {
        true
    }

    #[test]
    #[should_panic(expected = "requires label")]
    fn requires_label_works_with_run_if() {
        let mut scheduler = Scheduler::default();
        scheduler.suppress_warnings = true;
        scheduler.add_update_system(
            force_b
                .label("force_b")
                .requires_label("force_a")
                .run_if(always_true),
            ScheduleSet::Force,
        );
        scheduler.organize_systems();
    }

    #[test]
    fn requires_label_passes_with_run_if() {
        let mut scheduler = Scheduler::default();
        scheduler.suppress_warnings = true;
        scheduler.add_update_system(force_a.label("force_a"), ScheduleSet::Force);
        scheduler.add_update_system(
            force_b
                .label("force_b")
                .requires_label("force_a")
                .run_if(always_true),
            ScheduleSet::Force,
        );
        scheduler.organize_systems();
    }

    // ─── validate_schedule warning tests ─────────────────────────────────────

    #[test]
    fn validate_warns_no_force_systems() {
        let mut scheduler = Scheduler::default();
        scheduler.suppress_warnings = true;
        scheduler.add_update_system(force_a, ScheduleSet::InitialIntegration);
        scheduler.add_update_system(force_b, ScheduleSet::FinalIntegration);
        scheduler.organize_systems();
        let warnings = scheduler.schedule_warnings();
        assert!(
            warnings.iter().any(|w| w.contains("Force")),
            "Expected warning about missing Force systems, got: {:?}",
            warnings
        );
    }

    #[test]
    fn validate_warns_asymmetric_verlet() {
        let mut scheduler = Scheduler::default();
        scheduler.suppress_warnings = true;
        scheduler.add_update_system(force_a, ScheduleSet::InitialIntegration);
        scheduler.add_update_system(force_b, ScheduleSet::Force);
        scheduler.organize_systems();
        let warnings = scheduler.schedule_warnings();
        assert!(
            warnings.iter().any(|w| w.contains("asymmetric")),
            "Expected warning about asymmetric Verlet, got: {:?}",
            warnings
        );
    }

    #[test]
    fn validate_warns_no_integrator() {
        let mut scheduler = Scheduler::default();
        scheduler.suppress_warnings = true;
        fn sys_a() {}
        fn sys_b() {}
        fn sys_c() {}
        scheduler.add_update_system(sys_a, ScheduleSet::Force);
        scheduler.add_update_system(sys_b, ScheduleSet::PostForce);
        scheduler.add_update_system(sys_c, ScheduleSet::Exchange);
        scheduler.organize_systems();
        let warnings = scheduler.schedule_warnings();
        assert!(
            warnings.iter().any(|w| w.contains("no integrator")),
            "Expected warning about no integrator, got: {:?}",
            warnings
        );
    }

    #[test]
    fn validate_no_warnings_for_normal_schedule() {
        let mut scheduler = Scheduler::default();
        scheduler.suppress_warnings = true;
        scheduler.add_update_system(force_a, ScheduleSet::InitialIntegration);
        scheduler.add_update_system(force_b, ScheduleSet::Force);
        scheduler.add_update_system(force_a, ScheduleSet::FinalIntegration);
        scheduler.organize_systems();
        let warnings = scheduler.schedule_warnings();
        assert!(
            warnings.is_empty(),
            "Expected no warnings, got: {:?}",
            warnings
        );
    }

    // ─── function-handle-based ordering tests ─────────────────────────────────

    struct Counter(i32);

    fn sys_first(mut c: ResMut<Counter>) {
        assert_eq!(c.0, 0, "sys_first should run first");
        c.0 = 1;
    }

    fn sys_second(mut c: ResMut<Counter>) {
        assert_eq!(c.0, 1, "sys_second should run after sys_first");
        c.0 = 2;
    }

    #[test]
    fn fn_handle_after_ordering() {
        let mut scheduler = Scheduler::default();
        scheduler.suppress_warnings = true;
        scheduler.add_resource(Counter(0));
        // Register in reverse order — .after() should fix it
        scheduler.add_update_system(sys_second.after(sys_first), ScheduleSet::Force);
        scheduler.add_update_system(sys_first, ScheduleSet::Force);
        scheduler.add_scheduler_manager();
        scheduler.organize_systems();
        scheduler.run();
        let c = scheduler.get_resource_ref::<Counter>().unwrap();
        assert_eq!(c.0, 2);
    }

    #[test]
    fn fn_handle_before_ordering() {
        let mut scheduler = Scheduler::default();
        scheduler.suppress_warnings = true;
        scheduler.add_resource(Counter(0));
        // Register in reverse order — .before() should fix it
        scheduler.add_update_system(sys_second, ScheduleSet::Force);
        scheduler.add_update_system(sys_first.before(sys_second), ScheduleSet::Force);
        scheduler.add_scheduler_manager();
        scheduler.organize_systems();
        scheduler.run();
        let c = scheduler.get_resource_ref::<Counter>().unwrap();
        assert_eq!(c.0, 2);
    }

    #[test]
    fn fn_handle_requires_label_passes() {
        let mut scheduler = Scheduler::default();
        scheduler.suppress_warnings = true;
        scheduler.add_update_system(force_a, ScheduleSet::Force);
        scheduler.add_update_system(
            force_b.requires_label(force_a),
            ScheduleSet::Force,
        );
        scheduler.organize_systems();
    }

    #[test]
    #[should_panic(expected = "requires label")]
    fn fn_handle_requires_label_panics_when_missing() {
        let mut scheduler = Scheduler::default();
        scheduler.suppress_warnings = true;
        scheduler.add_update_system(
            force_b.requires_label(force_a),
            ScheduleSet::Force,
        );
        scheduler.organize_systems();
    }

    #[test]
    fn string_labels_still_work() {
        let mut scheduler = Scheduler::default();
        scheduler.suppress_warnings = true;
        scheduler.add_resource(Counter(0));
        scheduler.add_update_system(sys_second.after("first"), ScheduleSet::Force);
        scheduler.add_update_system(sys_first.label("first"), ScheduleSet::Force);
        scheduler.add_scheduler_manager();
        scheduler.organize_systems();
        scheduler.run();
        let c = scheduler.get_resource_ref::<Counter>().unwrap();
        assert_eq!(c.0, 2);
    }
}

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

// ─── Condition trait & FunctionCondition ─────────────────────────────────────

/// A DI-injected function that returns `bool`, used with `.run_if()`.
pub trait Condition {
    fn evaluate(&mut self, resources: &[RefCell<Box<dyn Any>>]) -> bool;
    fn prepare(&mut self, _index: &HashMap<TypeId, usize>) -> Vec<String> {
        Vec::new()
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
    pub fn before(mut self, target: impl Into<String>) -> Self {
        self.befores.push(target.into());
        self
    }
    pub fn after(mut self, target: impl Into<String>) -> Self {
        self.afters.push(target.into());
        self
    }
    pub fn requires_label(mut self, target: impl Into<String>) -> Self {
        self.requires.push(target.into());
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

    fn before(self, target: impl Into<String>) -> SystemDescriptor<Self::System> {
        SystemDescriptor {
            system: self.into_system(),
            label: None,
            befores: vec![target.into()],
            afters: vec![],
            requires: vec![],
        }
    }

    fn after(self, target: impl Into<String>) -> SystemDescriptor<Self::System> {
        SystemDescriptor {
            system: self.into_system(),
            label: None,
            befores: vec![],
            afters: vec![target.into()],
            requires: vec![],
        }
    }

    fn requires_label(self, target: impl Into<String>) -> SystemDescriptor<Self::System> {
        SystemDescriptor {
            system: self.into_system(),
            label: None,
            befores: vec![],
            afters: vec![],
            requires: vec![target.into()],
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
        StoredSystemEntry {
            system: Box::new(sys),
            name,
            label: None,
            befores: vec![],
            afters: vec![],
            requires: vec![],
        }
    }
}

/// ConditionalSystem — no ordering metadata.
impl<S: System + 'static, C: Condition + 'static> IntoScheduledSystem<CondMarker>
    for ConditionalSystem<S, C>
{
    fn into_stored(self) -> StoredSystemEntry {
        let name = self.name().to_string();
        StoredSystemEntry {
            system: Box::new(self),
            name,
            label: None,
            befores: vec![],
            afters: vec![],
            requires: vec![],
        }
    }
}

/// SystemDescriptor — carries label / before / after metadata.
impl<S: System + 'static> IntoScheduledSystem<DescMarker> for SystemDescriptor<S> {
    fn into_stored(self) -> StoredSystemEntry {
        let name = self.system.name().to_string();
        StoredSystemEntry {
            system: Box::new(self.system),
            name,
            label: self.label,
            befores: self.befores,
            afters: self.afters,
            requires: self.requires,
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

/// Run condition: returns true when the current state equals `target`.
pub fn in_state<S: Clone + PartialEq + 'static>(
    target: S,
) -> impl Fn(Res<CurrentState<S>>) -> bool {
    move |current: Res<CurrentState<S>>| current.0 == target
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

// ─── Topological sort within a ScheduleSet group ─────────────────────────────

fn topo_sort_group(group: &mut Vec<(StoredSystemEntry, ScheduleSet)>) {
    let n = group.len();
    if n <= 1 {
        return;
    }

    let mut label_to_idx: HashMap<String, usize> = HashMap::new();
    for (i, (entry, _)) in group.iter().enumerate() {
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

            // Validate requires_label: every required label must exist in this group
            let labels_in_group: Vec<Option<&str>> = group
                .iter()
                .map(|(entry, _)| entry.label.as_deref())
                .collect();
            for (entry, set) in &group {
                for req in &entry.requires {
                    if !labels_in_group.iter().any(|l| l == &Some(req.as_str())) {
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
        let has_initial = has_systems[set_to_value(&ScheduleSet::InitialIntegration) as usize];
        let has_final = has_systems[set_to_value(&ScheduleSet::FinalIntegration) as usize];
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
                self.organize_systems(); // systems may be added during setup in the future

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

    pub fn add_setup_system<I, S: System + 'static>(
        &mut self,
        system: impl IntoSystem<I, System = S>,
        schedule_set: ScheduleSetupSet,
    ) {
        let sys = system.into_system();
        let name = sys.name().to_string();
        self.setup_systems.push((
            StoredSystemEntry {
                system: Box::new(sys),
                name,
                label: None,
                befores: vec![],
                afters: vec![],
                requires: vec![],
            },
            schedule_set,
        ));
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

    fn short_name(full: &str) -> String {
        let parts: Vec<&str> = full.rsplitn(3, "::").collect();
        match parts.len() {
            0 => full.to_string(),
            1 => parts[0].to_string(),
            _ => format!("{}::{}", parts[1], parts[0]),
        }
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
            println!();
        }
        println!();
    }

    pub fn write_dot(&self, path: &str) {
        use std::io::Write;
        let mut out = String::new();
        out.push_str("digraph schedule {\n");
        out.push_str("    rankdir=TB;\n");
        out.push_str("    node [shape=box, style=filled, fillcolor=lightyellow];\n\n");

        // Helper to make valid DOT node IDs
        let node_id = |prefix: &str, idx: usize| -> String { format!("{}_{}", prefix, idx) };

        // Setup systems
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

        for (set_name, entries) in &setup_groups {
            out.push_str(&format!("    subgraph cluster_setup_{} {{\n", set_name));
            out.push_str(&format!("        label=\"Setup: {}\";\n", set_name));
            out.push_str("        style=filled; fillcolor=lightblue;\n");
            for &(i, entry) in entries {
                let short = Self::short_name(&entry.name);
                let label = if let Some(lbl) = &entry.label {
                    format!("{}\\n[{}]", short, lbl)
                } else {
                    short.to_string()
                };
                out.push_str(&format!(
                    "        {} [label=\"{}\"];\n",
                    node_id("setup", i),
                    label
                ));
            }
            out.push_str("    }\n\n");
        }

        // Edges between consecutive setup clusters (vertical layout)
        let setup_cluster_tails: Vec<String> = setup_groups
            .iter()
            .map(|(_, entries)| node_id("setup", entries.last().unwrap().0))
            .collect();
        let setup_cluster_heads: Vec<String> = setup_groups
            .iter()
            .map(|(_, entries)| node_id("setup", entries.first().unwrap().0))
            .collect();
        for i in 0..setup_cluster_tails.len().saturating_sub(1) {
            out.push_str(&format!(
                "    {} -> {} [color=blue, style=bold];\n",
                setup_cluster_tails[i],
                setup_cluster_heads[i + 1]
            ));
        }

        // Edges between systems within each setup cluster (vertical ordering)
        for (_set_name, entries) in &setup_groups {
            for w in entries.windows(2) {
                out.push_str(&format!(
                    "    {} -> {} [color=blue, style=bold];\n",
                    node_id("setup", w[0].0),
                    node_id("setup", w[1].0)
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
                let short = Self::short_name(&entry.name);
                let label = if let Some(lbl) = &entry.label {
                    format!("{}\\n[{}]", short, lbl)
                } else {
                    short.to_string()
                };
                out.push_str(&format!(
                    "        {} [label=\"{}\"];\n",
                    node_id("update", i),
                    label
                ));
            }
            out.push_str("    }\n\n");
        }

        // Edges between systems within each update cluster (vertical ordering)
        for (_set_name, entries) in &update_groups {
            for w in entries.windows(2) {
                out.push_str(&format!(
                    "    {} -> {} [color=blue, style=bold];\n",
                    node_id("update", w[0].0),
                    node_id("update", w[1].0)
                ));
            }
        }

        // Before/after constraint edges (red dashed)
        let mut label_to_update_node: HashMap<String, String> = HashMap::new();
        for (i, (entry, _)) in self.update_systems.iter().enumerate() {
            if let Some(lbl) = &entry.label {
                label_to_update_node.insert(lbl.clone(), node_id("update", i));
            }
        }

        for (i, (entry, _)) in self.update_systems.iter().enumerate() {
            let from = node_id("update", i);
            for b in &entry.befores {
                if let Some(target) = label_to_update_node.get(b) {
                    out.push_str(&format!(
                        "    {} -> {} [color=red, style=dashed, label=\"before\"];\n",
                        from, target
                    ));
                }
            }
            for a in &entry.afters {
                if let Some(source) = label_to_update_node.get(a) {
                    out.push_str(&format!(
                        "    {} -> {} [color=red, style=dashed, label=\"after\"];\n",
                        source, from
                    ));
                }
            }
        }

        // Blue edges between consecutive ScheduleSet clusters
        let cluster_tails: Vec<String> = update_groups
            .iter()
            .map(|(_, entries)| node_id("update", entries.last().unwrap().0))
            .collect();
        let cluster_heads: Vec<String> = update_groups
            .iter()
            .map(|(_, entries)| node_id("update", entries.first().unwrap().0))
            .collect();
        for i in 0..cluster_tails.len().saturating_sub(1) {
            out.push_str(&format!(
                "    {} -> {} [color=blue, style=bold];\n",
                cluster_tails[i],
                cluster_heads[i + 1]
            ));
        }

        // Loop-back edge from last update cluster to first (run loop)
        if cluster_tails.len() >= 2 {
            out.push_str(&format!(
                "    {} -> {} [color=green, style=bold, label=\"run loop\", constraint=false];\n",
                cluster_tails.last().unwrap(),
                cluster_heads.first().unwrap()
            ));
        }

        // Setup -> first update cluster edge
        if !setup_groups.is_empty() && !cluster_heads.is_empty() {
            let last_setup_group = setup_groups.last().unwrap();
            let last_setup_node = node_id("setup", last_setup_group.1.last().unwrap().0);
            out.push_str(&format!(
                "    {} -> {} [color=blue, style=bold, label=\"start run\"];\n",
                last_setup_node, cluster_heads[0]
            ));
        }

        out.push_str("}\n");

        let mut file = std::fs::File::create(path).expect("Failed to create DOT file");
        file.write_all(out.as_bytes())
            .expect("Failed to write DOT file");
        println!("Schedule DOT file written to: {}", path);
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
}

impl Default for SchedulerManager {
    fn default() -> Self {
        SchedulerManager {
            state: SchedulerState::Setup,
            index: 0,
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
        in_state,
        ConditionalSystem,
        // Simulation states
        CurrentState,
        IntoScheduledSystem,
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
}

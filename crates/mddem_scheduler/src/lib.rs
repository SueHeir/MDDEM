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
            fn run(&mut self, resources: &mut HashMap<TypeId, RefCell<Box<dyn Any>>>) {
                fn call_inner<$($params),*>(
                    mut f: impl FnMut($($params),*),
                    $($params: $params),*
                ) { f($($params),*) }

                let locals_ptr = &mut self.locals as *mut _;
                $(let $params = $params::retrieve(resources, locals_ptr);)*
                call_inner(&mut self.f, $($params),*)
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
                FunctionSystem { f: self, marker: Default::default(), locals: HashMap::new() }
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
            fn evaluate(&mut self, resources: &HashMap<TypeId, RefCell<Box<dyn Any>>>) -> bool {
                fn call_inner<$($params),*>(
                    mut f: impl FnMut($($params),*) -> bool,
                    $($params: $params),*
                ) -> bool { f($($params),*) }

                let locals_ptr = &mut self.locals as *mut _;
                $(let $params = $params::retrieve(resources, locals_ptr);)*
                call_inner(&mut self.f, $($params),*)
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
                FunctionCondition { f: self, marker: Default::default(), locals: HashMap::new() }
            }
        }
    }
}

// ─── SystemParam ──────────────────────────────────────────────────────────────

// ANCHOR: SystemParam
pub trait SystemParam {
    type Item<'new>;
    fn retrieve<'r>(
        resources: &'r HashMap<TypeId, RefCell<Box<dyn Any>>>,
        locals: *mut HashMap<TypeId, Box<dyn Any>>,
    ) -> Self::Item<'r>;
}
// ANCHOR_END: SystemParam

// ANCHOR: ResSystemParam
impl<'res, T: 'static> SystemParam for Res<'res, T> {
    type Item<'new> = Res<'new, T>;
    fn retrieve<'r>(
        resources: &'r HashMap<TypeId, RefCell<Box<dyn Any>>>,
        _locals: *mut HashMap<TypeId, Box<dyn Any>>,
    ) -> Self::Item<'r> {
        Res { value: resources.get(&TypeId::of::<T>()).unwrap().borrow(), _marker: PhantomData }
    }
}
// ANCHOR_END: ResSystemParam

// ANCHOR: ResMutSystemParam
impl<'res, T: 'static> SystemParam for ResMut<'res, T> {
    type Item<'new> = ResMut<'new, T>;
    fn retrieve<'r>(
        resources: &'r HashMap<TypeId, RefCell<Box<dyn Any>>>,
        _locals: *mut HashMap<TypeId, Box<dyn Any>>,
    ) -> Self::Item<'r> {
        ResMut { value: resources.get(&TypeId::of::<T>()).unwrap().borrow_mut(), _marker: PhantomData }
    }
}
// ANCHOR_END: ResMutSystemParam

// ─── Res / ResMut / Local ─────────────────────────────────────────────────────

// ANCHOR: Res
pub struct Res<'a, T: 'static> {
    value: Ref<'a, Box<dyn Any>>,
    _marker: PhantomData<&'a T>,
}
// ANCHOR_END: Res

impl<T: 'static> Deref for Res<'_, T> {
    type Target = T;
    fn deref(&self) -> &T { self.value.downcast_ref().unwrap() }
}

// ANCHOR: ResMut
pub struct ResMut<'a, T: 'static> {
    value: RefMut<'a, Box<dyn Any>>,
    _marker: PhantomData<&'a mut T>,
}

impl<T: 'static> Deref for ResMut<'_, T> {
    type Target = T;
    fn deref(&self) -> &T { self.value.downcast_ref().unwrap() }
}
impl<T: 'static> DerefMut for ResMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T { self.value.downcast_mut().unwrap() }
}
// ANCHOR_END: ResMut

// ANCHOR: Local
/// Per-system local state. Persists across invocations of the same system instance.
/// Initialized with `T::default()` on first access.
pub struct Local<'a, T: Default + 'static> {
    value: &'a mut T,
    _marker: PhantomData<&'a mut T>,
}

impl<'res, T: Default + 'static> SystemParam for Local<'res, T> {
    type Item<'new> = Local<'new, T>;
    fn retrieve<'r>(
        _resources: &'r HashMap<TypeId, RefCell<Box<dyn Any>>>,
        locals: *mut HashMap<TypeId, Box<dyn Any>>,
    ) -> Self::Item<'r> {
        // SAFETY: locals points to FunctionSystem::locals, exclusively owned by this
        // system and alive for the duration of this retrieve call.
        let map = unsafe { &mut *locals };
        let entry = map.entry(TypeId::of::<T>()).or_insert_with(|| Box::new(T::default()));
        Local { value: entry.downcast_mut::<T>().unwrap(), _marker: PhantomData }
    }
}

impl<T: Default + 'static> Deref for Local<'_, T> {
    type Target = T;
    fn deref(&self) -> &T { self.value }
}
impl<T: Default + 'static> DerefMut for Local<'_, T> {
    fn deref_mut(&mut self) -> &mut T { self.value }
}
// ANCHOR_END: Local

// ─── System trait & FunctionSystem ───────────────────────────────────────────

// ANCHOR: System
pub trait System {
    fn run(&mut self, resources: &mut HashMap<TypeId, RefCell<Box<dyn Any>>>);
    fn name(&self) -> &str { "unknown" }
}
// ANCHOR_END: System

pub struct FunctionSystem<Input, F> {
    f: F,
    marker: PhantomData<fn() -> Input>,
    /// Per-system-instance local state, keyed by TypeId.
    locals: HashMap<TypeId, Box<dyn Any>>,
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
    fn evaluate(&mut self, resources: &HashMap<TypeId, RefCell<Box<dyn Any>>>) -> bool;
}

pub struct FunctionCondition<Input, F> {
    f: F,
    marker: PhantomData<fn() -> Input>,
    locals: HashMap<TypeId, Box<dyn Any>>,
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
    fn run(&mut self, resources: &mut HashMap<TypeId, RefCell<Box<dyn Any>>>) {
        if self.condition.evaluate(resources) {
            self.system.run(resources);
        }
    }
    fn name(&self) -> &str { self.system.name() }
}

// ─── SystemDescriptor ─────────────────────────────────────────────────────────

/// Wraps a system with ordering metadata (label, before/after constraints).
pub struct SystemDescriptor<S: System + 'static> {
    pub system: S,
    pub label: Option<String>,
    pub befores: Vec<String>,
    pub afters: Vec<String>,
}

impl<S: System + 'static> SystemDescriptor<S> {
    pub fn label(mut self, lbl: impl Into<String>) -> Self {
        self.label = Some(lbl.into()); self
    }
    pub fn before(mut self, target: impl Into<String>) -> Self {
        self.befores.push(target.into()); self
    }
    pub fn after(mut self, target: impl Into<String>) -> Self {
        self.afters.push(target.into()); self
    }
    pub fn run_if<I2, C: Condition + 'static>(self, cond: impl IntoCondition<I2, Condition = C>)
        -> SystemDescriptor<ConditionalSystem<S, C>>
    {
        SystemDescriptor {
            system: ConditionalSystem { system: self.system, condition: cond.into_condition() },
            label: self.label,
            befores: self.befores,
            afters: self.afters,
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
    fn run_if<I2, C: Condition + 'static>(self, cond: impl IntoCondition<I2, Condition = C>)
        -> ConditionalSystem<Self::System, C>
    {
        ConditionalSystem { system: self.into_system(), condition: cond.into_condition() }
    }

    fn label(self, lbl: impl Into<String>) -> SystemDescriptor<Self::System> {
        SystemDescriptor { system: self.into_system(), label: Some(lbl.into()), befores: vec![], afters: vec![] }
    }

    fn before(self, target: impl Into<String>) -> SystemDescriptor<Self::System> {
        SystemDescriptor { system: self.into_system(), label: None, befores: vec![target.into()], afters: vec![] }
    }

    fn after(self, target: impl Into<String>) -> SystemDescriptor<Self::System> {
        SystemDescriptor { system: self.into_system(), label: None, befores: vec![], afters: vec![target.into()] }
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
        StoredSystemEntry { system: Box::new(sys), name, label: None, befores: vec![], afters: vec![] }
    }
}

/// ConditionalSystem — no ordering metadata.
impl<S: System + 'static, C: Condition + 'static> IntoScheduledSystem<CondMarker>
    for ConditionalSystem<S, C>
{
    fn into_stored(self) -> StoredSystemEntry {
        let name = self.name().to_string();
        StoredSystemEntry { system: Box::new(self), name, label: None, befores: vec![], afters: vec![] }
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
}

// ─── Schedule sets ────────────────────────────────────────────────────────────

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
    pub fn set(&mut self, state: S) { self.0 = Some(state); }
    pub fn clear(&mut self) { self.0 = None; }
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
    if n <= 1 { return; }

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
            if in_degree[nbr] == 0 { queue.push_back(nbr); }
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
pub struct Scheduler {
    setup_systems: Vec<(StoredSystemEntry, ScheduleSetupSet)>,
    update_systems: Vec<(StoredSystemEntry, ScheduleSet)>,
    pub resources: HashMap<TypeId, RefCell<Box<dyn Any>>>,
    print_schedule: bool,
}
// ANCHOR_END: Scheduler

impl Default for Scheduler {
    fn default() -> Self {
        Scheduler {
            setup_systems: Vec::new(),
            update_systems: Vec::new(),
            resources: HashMap::new(),
            print_schedule: false,
        }
    }
}

// ANCHOR: SchedulerImpl
impl Scheduler {
    pub fn organize_systems(&mut self) {
        self.setup_systems.sort_by_key(|(_, f)| setup_set_to_value(f));
        self.update_systems.sort_by_key(|(_, f)| set_to_value(f));

        // Topo sort within each ScheduleSet group
        let all = std::mem::take(&mut self.update_systems);
        if all.is_empty() { return; }

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

        for mut group in groups {
            topo_sort_group(&mut group);
            self.update_systems.extend(group);
        }
    }

    pub fn setup(&mut self) {
        for (entry, _set) in self.setup_systems.iter_mut() {
            entry.system.run(&mut self.resources);
        }
    }

    pub fn run(&mut self) {
        for (entry, _set) in self.update_systems.iter_mut() {
            entry.system.run(&mut self.resources);
        }
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

                let mut binding = self.resources
                    .get(&TypeId::of::<SchedulerManager>()).unwrap().borrow_mut();
                let sm = binding.downcast_mut::<SchedulerManager>().unwrap();
                sm.state = SchedulerState::Run;
            }

            if matches!(schedule_state, SchedulerState::Run) {
                self.run();
            }

            let mut binding = self.resources
                .get(&TypeId::of::<SchedulerManager>()).unwrap().borrow_mut();
            let sm = binding.downcast_mut::<SchedulerManager>().unwrap();
            schedule_state = sm.state.clone();
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
                label: None, befores: vec![], afters: vec![],
            },
            schedule_set,
        ));
    }

    pub fn add_update_system<M>(
        &mut self,
        system: impl IntoScheduledSystem<M>,
        schedule_set: ScheduleSet,
    ) {
        self.update_systems.push((system.into_stored(), schedule_set));
    }

    pub fn add_scheduler_manager(&mut self) {
        self.add_resource(SchedulerManager::new());
    }

    pub fn add_resource<R: 'static>(&mut self, res: R) {
        self.resources.insert(TypeId::of::<R>(), RefCell::new(Box::new(res)));
    }

    pub fn get_mut_resource(&mut self, res: TypeId) -> Option<&RefCell<Box<dyn Any>>> {
        self.resources.get(&res)
    }

    pub fn get_resource_ref<R: 'static>(&self) -> Option<std::cell::Ref<'_, R>> {
        self.resources.get(&TypeId::of::<R>()).map(|cell| {
            std::cell::Ref::map(cell.borrow(), |b| b.downcast_ref::<R>().unwrap())
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
        println!("\n{}", "═══ Setup Systems ═══");
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

        println!("\n{}", "═══ Update Systems (per-step) ═══");
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
        let node_id = |prefix: &str, idx: usize| -> String {
            format!("{}_{}", prefix, idx)
        };

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
                out.push_str(&format!("        {} [label=\"{}\"];\n", node_id("setup", i), label));
            }
            out.push_str("    }\n\n");
        }

        // Edges between consecutive setup clusters (vertical layout)
        let setup_cluster_tails: Vec<String> = setup_groups.iter()
            .map(|(_, entries)| node_id("setup", entries.last().unwrap().0))
            .collect();
        let setup_cluster_heads: Vec<String> = setup_groups.iter()
            .map(|(_, entries)| node_id("setup", entries.first().unwrap().0))
            .collect();
        for i in 0..setup_cluster_tails.len().saturating_sub(1) {
            out.push_str(&format!(
                "    {} -> {} [color=blue, style=bold];\n",
                setup_cluster_tails[i], setup_cluster_heads[i + 1]
            ));
        }

        // Edges between systems within each setup cluster (vertical ordering)
        for (_set_name, entries) in &setup_groups {
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
                let short = Self::short_name(&entry.name);
                let label = if let Some(lbl) = &entry.label {
                    format!("{}\\n[{}]", short, lbl)
                } else {
                    short.to_string()
                };
                out.push_str(&format!("        {} [label=\"{}\"];\n", node_id("update", i), label));
            }
            out.push_str("    }\n\n");
        }

        // Edges between systems within each update cluster (vertical ordering)
        for (_set_name, entries) in &update_groups {
            for w in entries.windows(2) {
                out.push_str(&format!(
                    "    {} -> {} [color=blue, style=bold];\n",
                    node_id("update", w[0].0), node_id("update", w[1].0)
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
        let cluster_tails: Vec<String> = update_groups.iter()
            .map(|(_, entries)| node_id("update", entries.last().unwrap().0))
            .collect();
        let cluster_heads: Vec<String> = update_groups.iter()
            .map(|(_, entries)| node_id("update", entries.first().unwrap().0))
            .collect();
        for i in 0..cluster_tails.len().saturating_sub(1) {
            out.push_str(&format!(
                "    {} -> {} [color=blue, style=bold];\n",
                cluster_tails[i], cluster_heads[i + 1]
            ));
        }

        // Loop-back edge from last update cluster to first (run loop)
        if cluster_tails.len() >= 2 {
            out.push_str(&format!(
                "    {} -> {} [color=green, style=bold, label=\"run loop\", constraint=false];\n",
                cluster_tails.last().unwrap(), cluster_heads.first().unwrap()
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
        file.write_all(out.as_bytes()).expect("Failed to write DOT file");
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

pub struct SchedulerManager {
    pub state: SchedulerState,
    pub index: usize,
}

impl SchedulerManager {
    pub fn new() -> Self {
        SchedulerManager { state: SchedulerState::Setup, index: 0 }
    }
}

// ─── Prelude ──────────────────────────────────────────────────────────────────

pub mod prelude {
    pub use crate::{
        // Core DI
        Scheduler, Res, ResMut, Local,
        // System scheduling
        ScheduleSet, ScheduleSetupSet, SchedulerManager, SchedulerState,
        // Run conditions
        SystemExt, ConditionalSystem,
        // System ordering
        SystemDescriptor, IntoScheduledSystem,
        // Simulation states
        CurrentState, NextState, in_state, apply_state_transitions,
    };
}

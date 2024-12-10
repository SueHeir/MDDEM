// ANCHOR: All
use std::any::{Any, TypeId};
use std::cell::{Ref, RefCell, RefMut};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

macro_rules! impl_system {
    (
        $($params:ident),*
    ) => {
        #[allow(non_snake_case)]
        #[allow(unused)]
        impl<F, $($params: SystemParam),*> System for FunctionSystem<($($params,)*), F>
            where
                for<'a, 'b> &'a mut F:
                    FnMut( $($params),* ) +
                    FnMut( $(<$params as SystemParam>::Item<'b>),* )
        {
            fn run(&mut self, resources: &mut HashMap<TypeId, RefCell<Box<dyn Any>>>) {
                fn call_inner<$($params),*>(
                    mut f: impl FnMut($($params),*),
                    $($params: $params),*
                ) {
                    f($($params),*)
                }

                $(
                    let $params = $params::retrieve(resources);
                )*

                call_inner(&mut self.f, $($params),*)
            }
        }
    }
}

macro_rules! impl_into_system {
    (
        $($params:ident),*
    ) => {
        impl<F, $($params: SystemParam),*> IntoSystem<($($params,)*)> for F
            where
                for<'a, 'b> &'a mut F:
                    FnMut( $($params),* ) +
                    FnMut( $(<$params as SystemParam>::Item<'b>),* )
        {
            type System = FunctionSystem<($($params,)*), Self>;

            fn into_system(self) -> Self::System {
                FunctionSystem {
                    f: self,
                    marker: Default::default(),
                }
            }
        }
    }
}

// ANCHOR: SystemParam
pub trait SystemParam {
    type Item<'new>;

    fn retrieve<'r>(resources: &'r HashMap<TypeId, RefCell<Box<dyn Any>>>) -> Self::Item<'r>;
}
// ANCHOR_END: SystemParam

// ANCHOR: ResSystemParam
impl<'res, T: 'static> SystemParam for Res<'res, T> {
    type Item<'new> = Res<'new, T>;

    fn retrieve<'r>(resources: &'r HashMap<TypeId, RefCell<Box<dyn Any>>>) -> Self::Item<'r> {
        Res {
            value: resources.get(&TypeId::of::<T>()).unwrap().borrow(),
            _marker: PhantomData,
        }
    }
}
// ANCHOR_END: ResSystemParam

// ANCHOR: ResMutSystemParam
impl<'res, T: 'static> SystemParam for ResMut<'res, T> {
    type Item<'new> = ResMut<'new, T>;

    fn retrieve<'r>(resources: &'r HashMap<TypeId, RefCell<Box<dyn Any>>>) -> Self::Item<'r> {
        ResMut {
            value: resources.get(&TypeId::of::<T>()).unwrap().borrow_mut(),
            _marker: PhantomData,
        }
    }
}
// ANCHOR_END: ResMutSystemParam

// ANCHOR: Res
pub struct Res<'a, T: 'static> {
    value: Ref<'a, Box<dyn Any>>,
    _marker: PhantomData<&'a T>,
}
// ANCHOR_END: Res

// ANCHOR: ResDeref
impl<T: 'static> Deref for Res<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value.downcast_ref().unwrap()
    }
}
// ANCHOR_END: ResDeref

// ANCHOR: ResMut
pub struct ResMut<'a, T: 'static> {
    value: RefMut<'a, Box<dyn Any>>,
    _marker: PhantomData<&'a mut T>,
}

impl<T: 'static> Deref for ResMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value.downcast_ref().unwrap()
    }
}

impl<T: 'static> DerefMut for ResMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.downcast_mut().unwrap()
    }
}
// ANCHOR_END: ResMut
#[derive(Debug)]
pub enum ScheduleSet {
    Setup,
    PreInitalIntegration,
    InitalIntegration,
    PostInitalIntegration,
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

pub fn set_to_value(schedule_set: &ScheduleSet) -> u32 {
    match schedule_set {
        ScheduleSet::Setup => 0,
        ScheduleSet::PreInitalIntegration => 1,
        ScheduleSet::InitalIntegration => 2,
        ScheduleSet::PostInitalIntegration => 3,
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

pub struct FunctionSystem<Input, F> {
    f: F,

    marker: PhantomData<fn() -> Input>,
}

// ANCHOR: System

pub trait System {
    fn run(&mut self, resources: &mut HashMap<TypeId, RefCell<Box<dyn Any>>>);
}
// ANCHOR_END: System

impl_system!();
impl_system!(T1);
impl_system!(T1, T2);
impl_system!(T1, T2, T3);
impl_system!(T1, T2, T3, T4);
impl_system!(T1, T2, T3, T4, T5);

pub trait IntoSystem<Input> {
    type System: System;

    fn into_system(self) -> Self::System;
}

impl_into_system!();
impl_into_system!(T1);
impl_into_system!(T1, T2);
impl_into_system!(T1, T2, T3);
impl_into_system!(T1, T2, T3, T4);
impl_into_system!(T1, T2, T3, T4, T5);

type StoredSystem = Box<dyn System>;

// ANCHOR: Scheduler
#[derive(Default)]
pub struct Scheduler {
    setup_systems: Vec<(StoredSystem, ScheduleSet)>,
    update_systems: Vec<(StoredSystem, ScheduleSet)>,
    pub resources: HashMap<TypeId, RefCell<Box<dyn Any>>>,
}
// ANCHOR_END: Scheduler

// ANCHOR: SchedulerImpl
impl Scheduler {
    pub fn organize_systems(&mut self) {
        self.setup_systems.sort_by_key(|(_, f)| set_to_value(f));
        self.update_systems.sort_by_key(|(_, f)| set_to_value(f));
    }

    pub fn setup(&mut self) {
        for (system, _set) in self.setup_systems.iter_mut() {
            // println!("setup {:?}:", _set);
            system.run(&mut self.resources);
        }
    }
    pub fn run(&mut self) {
        for (system, _set) in self.update_systems.iter_mut() {
            // println!("run {:?}:", _set);
            system.run(&mut self.resources);
        }
    }

    pub fn start(&mut self) {
        self.add_scheduler_manager();
        let mut schedule_state = SchedulerState::Setup;
        while !matches!(schedule_state, SchedulerState::End) {
            
            if matches!(schedule_state, SchedulerState::Setup) {
                self.organize_systems();
                self.setup();
                self.organize_systems(); //In the future we will be able to add or remove systems during setup!
                
                let mut binding = self
                .resources
                .get(&TypeId::of::<SchedulerManager>())
                .unwrap()
                .borrow_mut();
                let scheduler_manager = binding.downcast_mut::<SchedulerManager>().unwrap();
                scheduler_manager.state = SchedulerState::Run;
            }

            if matches!(schedule_state, SchedulerState::Run) {
                self.run();
            }


            //get Data from scheduleManager
            
            let mut binding = self
                .resources
                .get(&TypeId::of::<SchedulerManager>())
                .unwrap()
                .borrow_mut();
            let scheduler_manager = binding.downcast_mut::<SchedulerManager>().unwrap();
            schedule_state = scheduler_manager.state.clone();
            
        }
        
    }

    pub fn add_setup_system<I, S: System + 'static>(
        &mut self,
        system: impl IntoSystem<I, System = S>,
        schedule_set: ScheduleSet,
    ) {
        self.setup_systems
            .push((Box::new(system.into_system()), schedule_set));
    }

    pub fn add_update_system<I, S: System + 'static>(
        &mut self,
        system: impl IntoSystem<I, System = S>,
        schedule_set: ScheduleSet,
    ) {
        self.update_systems
            .push((Box::new(system.into_system()), schedule_set));
    }

    pub fn add_scheduler_manager(&mut self){
        self.add_resource(SchedulerManager::new());
    }

    pub fn add_resource<R: 'static>(&mut self, res: R) {
        self.resources
            .insert(TypeId::of::<R>(), RefCell::new(Box::new(res)));
    }
    pub fn get_mut_resource(&mut self, res: TypeId) -> Option<&RefCell<Box<dyn Any>>>{
        let binding: Option<&RefCell<Box<dyn Any>>> = self
                .resources
                .get(&res);
        return binding;
    }
    
}
// ANCHOR_END: SchedulerImpl
// ANCHOR_END: All


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


pub mod prelude {
    pub use crate::{
        Scheduler,
        ScheduleSet,
        SchedulerManager,
        SchedulerState,
        Res,
        ResMut,
    };
}

# MDDEM: Molecular Dynamics - Discrete Element Method
MDDEM (Pronouced Madem without the 'a' sound), is a MPI parallelized Molecular Dynamics or Discrete Element Method Codebase. 
The codebase uses [rsmpi](https://github.com/rsmpi/rsmpi) as a rust wrapper around the MPI protocal and [nalgebra](https://github.com/dimforge/nalgebra) as a math library.

Currently this is a minimal running project with a LAMMPS or LIGGGHTS style MPI communication. Exchange, Borders, and Reverse communcation are currently working
for the simple simulation of a perodic box of spheres contacting eachother with a hertz normal force. The input script below runs such simulation where the path to the input script is taken
as the first argument while running MDDEM.

```
processors        2 2 1
neighbor          1.1
domain            0.0 0.02 0.0 0.02 0.0 0.01
periodic          p p p
gravity           0.0 -9.8 0.0


randomparticleinsert    150 0.001 2500 8.7e9 0.3
randomparticlevelocity  0.05
dampening               0.95
run                     1000000
```

```bash
mpiexec -n 4 ./target/release/MDDEM ./input
```

## Code Layout
The goal of MDDEM is to be modular and easily editable.  Currently all function calls during the simulation take place via a scheduler that hold structs of data (Domain, Atoms, Comm) which can then
be injected into any function via [dependecy injection](https://github.com/PROMETHIA-27/dependency_injection_like_bevy_from_scratch/blob/main/src/chapter3/interior_mutability.md). 
The scheduler currently calls functions based on the following ScheduleSet enum in order (very similar to LAMMPS and LIGGGHTS).

```rust
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
```

Functions and structs are registered to the scheduler as follows

```rust
pub fn comm_app(scheduler: &mut Scheduler) {
    scheduler.add_resource(Comm::new());
    scheduler.add_setup_system(read_input, Setup);
    scheduler.add_setup_system(setup, PreNeighbor);

    scheduler.add_update_system(exchange, Exchange);
    scheduler.add_update_system(borders, PreNeighbor);
    scheduler.add_update_system(reverse_send_force, PostForce);
}
```


## Future Goals
- Modularity through a plugin manager
  - Everything should be a plugin that can be easily removed similar to that of [bevy](https://github.com/bevyengine/bevy).
  - A specific type of simulation will require someone to build their own version of MDDEM, using either internal or external plugins, with the plugins of their choice (That is why it can be both a MD and DEM codebase)
    - This should be as easy as it is to start a new bevy project, add MDDEM to your .toml file, add your LEBC plugin to your .toml file, compose your MDDEM to use the function that you need.
    - This will lead to simpler code written as one function doesn't need to handle every kind of DEM or MD simulation there is.
  - Some functions calls (like a modified LEBC borders function) will also require different accompanying functions. There should be a compile-time check that each function used has all the requires accompanying function
  - Although the plugin manager and scheduler will end up similar to bevy, the goal is not to become an ECS based codebase as I believe this complicated many things unnesessary.
- A Real world Application for both a DEM Project and MD Project
  - DEM example will most likely be some form of LEBC implimented for single spheres
  - MD example is not yet determined


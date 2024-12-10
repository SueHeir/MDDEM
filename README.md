# MDDEM: Molecular Dynamics - Discrete Element Method
MDDEM (Pronouced Madem without the 'a' sound), is a MPI parallelized Molecular Dynamics or Discrete Element Method Codebase. 
The codebase uses [rsmpi](https://github.com/rsmpi/rsmpi) as a rust wrapper around the MPI protocal and [nalgebra](https://github.com/dimforge/nalgebra) as a math library.

Currently this is a minimal running project with a LAMMPS or LIGGGHTS style MPI communication. Exchange, Borders, and Reverse communcation are currently working
for the simple simulation of a perodic box of spheres contacting each other with a hertz normal force. The input script below runs such simulation where the path to the input script is taken
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
The goal of MDDEM is to be modular and easily editable.  Currently all function calls during the simulation take place via a scheduler that holds structs of data (Domain, Atoms, Comm) which can then
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

Functions and structs are registered to the scheduler with a Plugin. Plugins' build function is called when the plugin is added to the application.

```rust
pub struct CommincationPlugin;

impl Plugin for CommincationPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Comm::new())
            .add_setup_system(read_input, ScheduleSet::Setup)
            .add_setup_system(setup, ScheduleSet::PreNeighbor)
            .add_update_system(exchange, ScheduleSet::Exchange)
            .add_update_system(borders, ScheduleSet::PreNeighbor)
            .add_update_system(reverse_send_force, ScheduleSet::PostForce);
    }
}
```

Currently this project is not set up to be a library but will be in the future (Probably when I add it to cargo). For now the main function composes a runnable version of the codebase through adding all the required plugins.
```rust
fn main() {
   App::new()
        .add_plugins(InputPlugin)
        .add_plugins(CommincationPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(NeighborPlugin)
        .add_plugins(DemAtomPlugin)
        .add_plugins(ForcePlugin)
        .add_plugins(VerletPlugin)
        .add_plugins(PrintPlugin)
        .start();

}
```
The Atom plugin does not need added here because DemAtomPlugin adds it.  Atom contains an added value which holds of vector of pointers to additionally added data to use throughout the simulation. For example, the young's modulus of each atom is stored in the DemAtom struct held by atom's added value. All structs added to atom have the AtomAdded trait which defines functions to get and recieve buffers for mpi communication.


## TODO
- Adding plugins in the wrong order can break it, this needs fixed
- Addable and Removable systems during setup
- Check that each function used has all the required accompanying functions
- MPI forward and reverse communcation for any unknown struct
  - This should allow them to be reused for any type of communication

## Future Goals
- Modularity through a plugin manager (partially done)
  - Everything should be a plugin that can be easily removed similar to that of [bevy](https://github.com/bevyengine/bevy).
  - A specific type of simulation will require someone to build their own version of MDDEM, using either internal or external plugins, with the plugins of their choice (That is why it can be both a MD and DEM codebase)
    - This should be as easy as it is to start a new bevy project: add MDDEM to your .toml file, add your LEBC plugin to your .toml file, and compose your MDDEM to use the function that you need.
    - This will lead to simpler code written as one function doesn't need to handle every kind of DEM or MD simulation there is.
  - Some function calls (like a modified LEBC borders function) will also require different accompanying functions. There should be a compile-time check that each function used has all the required accompanying functions
- A real world application for both a DEM project and MD project
  - DEM example will most likely be some form of LEBC implimented for single spheres
  - MD example is not yet determined


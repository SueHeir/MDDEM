# Why NASA and the Scientific Community Should Adopt a Shared Scheduler/App Framework in Rust

## Core Thesis

Scientific computing is entering an era where new codebases are being written faster than ever (AI-assisted development), but the coupling and interoperability problem is getting worse, not better. A shared scheduler/app framework in Rust would solve this at the foundation level.

## Case Study: MDDEM

This repository implements approximately 20% of LAMMPS's core MD feature set and 90% of its DEM feature set, all built on the proposed scheduler architecture. It was developed in two weeks of part-time work using AI-assisted coding.

While much of this code has not been rigorously validated — there are bugs and rough edges in the developer experience — it demonstrates how a scheduler/plugin system keeps code clean, organized, and modifiable at scale. If new codebases adopted this shared scheduler pattern, coupling between them would become trivial from a data transfer, serialization, and synchronization standpoint. The physics is still the scientist's job — the framework just makes sure we stop spending our time on plumbing.

### Performance

Although these benchmarks have not been re-run recently, MDDEM achieved 97% of LAMMPS performance for a Lennard-Jones fluid on a single core, and 90% for MPI-parallel runs. Two weeks versus thirty years of development — and yes, Claude is standing on the shoulders of open-source giants here. But the point stands: why optimize for a 10-day simulation versus an 11-day simulation, when coupling two codes the traditional way costs months of engineering time?

---

## A) Why Rust

### Safety Without Sacrificing Performance
- No segfaults, no data races, no buffer overflows — enforced at compile time (these scientific codes use unsafe code all the time for performance, but you know exeactly where the unsafe code is)
- These bugs are the #1 time sink in debugging Fortran/C++ scientific codes

### Performance Parity with C/C++
- Zero-cost abstractions — iterators, generics, and traits compile to the same assembly as hand-written C
- No garbage collector, no runtime overhead
- LLVM backend — same optimization passes as Clang

### Fearless Concurrency
- The borrow checker prevents data races at compile time
- Critical as codes move toward GPU offloading, async I/O, and hybrid MPI+threads
- Fortran and C++ data races are notoriously difficult to find and reproduce

### Modern Tooling
- `cargo` handles builds, dependencies, testing, and benchmarking — no more CMake nightmares
- Built-in package manager makes reusing code across projects trivial
- `cargo test`, `cargo bench`, `cargo doc` — testing and documentation are first-class citizens

### Longevity and Maintainability
- A strong type system makes refactoring safe — the compiler catches breakage before runtime
- Code written by one person or team is readable and modifiable by others
- Consider the difference: inheriting a 200k-line Fortran codebase versus a 200k-line Rust codebase

### Growing Ecosystem
- MPI bindings (`mpi` crate), linear algebra (`nalgebra`, `faer`), HDF5, and VTK output libraries already exist
- DOE and national labs are actively exploring Rust for scientific computing

---

## B) Why a Shared Scheduler / App Framework

### The Coupling Problem Is the Expensive Problem
- Building a single-physics solver is relatively straightforward
- Coupling two solvers is where projects burn years of engineering time
- CFDEM, MOOSE+BISON, OpenFOAM+LIGGGHTS — all required enormous effort on integration glue code
- A shared scheduler reduces coupling to a plugin registration problem, not an integration project

### Zero-Copy Interoperability by Construction
- Same-process, same-address-space data sharing between physics modules
- No serialization, no file-based coupling, no MPI inter-communicators
- DEM reads fluid fields directly; CFD reads particle positions directly
- Orders of magnitude less coupling overhead than traditional approaches

### Explicit Data Dependencies
- `Res<T>` (read) and `ResMut<T>` (write) make data flow visible and compiler-checked
- No hidden global state mutations — if a system modifies mesh data, it declares so in the function signature
- Makes it possible to reason about correctness of coupled physics

### Subcycling and Multi-Rate Integration Become Trivial
- Different physics modules can run at different rates using `run_if` conditions
- No external synchronization protocol needed
- The scheduler handles ordering; physics developers focus on physics

### Plugin Architecture Enables Modularity
- Each physics capability is a self-contained plugin
- Teams can develop independently, test independently, and compose at runtime
- Swap Hertz contact for JKR? Change one plugin. Replace k-epsilon with LES? One plugin.
- This is how modern software works (VS Code, game engines) — scientific computing should catch up

### Standardized Phase Ordering
- Even when codes are not coupled, a shared phase vocabulary (setup, integrate, exchange, compute forces, finalize) reduces cognitive overhead
- New developers understand where their code fits immediately

---

## C) Reducing Boilerplate and Learning Curves

### AI Is Accelerating New Codebases — But Not Coupling
- An LLM can generate a standalone DEM solver or CFD solver in days
- But AI cannot retroactively make two independently-written codes interoperate
- If both started from the same framework, coupling is already solved

### Scientists Should Write Physics, Not Infrastructure
- Config file parsing, output formatting, restart files, MPI boilerplate, neighbor lists, domain decomposition — these are solved problems
- Every new scientific code re-implements them slightly differently
- A shared framework provides these once, correctly, and tested

### One Learning Curve Instead of N (I would like to stress this)
- Learn the scheduler/plugin/resource pattern once; apply it to DEM, CFD, FEM, radiation transport, chemistry, etc.
- Currently: learn LAMMPS's fix/compute/pair architecture, AND OpenFOAM's fvMesh/fvSchemes, AND MOOSE's Kernel/Material system — all different, all serving the same purpose
- A shared framework means transferable knowledge across domains

### Onboarding and Workforce Development (I would like to stress this)
- New hires, interns, and postdocs become productive faster
- "Here's how you write a plugin" is a 30-minute tutorial, not a 6-month apprenticeship
- Code review is easier when everyone speaks the same architectural language

### Testing and Validation
- Shared infrastructure means shared test patterns
- Unit test a physics plugin in isolation; integration test the composition
- CI/CD pipelines work identically across all codes built on the framework

---

## D) Strategic and Organizational Arguments for NASA

### Verification & Validation Consistency
- A shared framework means V&V procedures transfer across codes
- Easier to certify when the infrastructure layer is common and well-tested

### Reducing Bus Factor
- Many NASA codes are maintained by 1-3 people
- A shared framework means other teams can understand and contribute to any code built on it
- When someone retires or leaves, their code is not an opaque monolith

### Faster Response to New Mission Needs
- Need DEM-CFD coupling for lunar regolith plume interaction? Plug them together.
- Need radiation transport coupled to thermal? Same pattern.
- The framework turns a "multi-year integration project" into "multi-week plugin development"

### Open-Source Community Leverage (I would like to stress this)
- A well-designed shared framework attracts external contributors
- Universities, national labs, and international partners can all build plugins
- NASA maintains the core; the community extends it

### AI-Assisted Development Works Better with Structure
- LLMs are dramatically better at generating code that follows a clear pattern or template
- "Write a plugin that implements X" is a well-constrained prompt
- Unstructured Fortran codebases are significantly harder for AI to extend correctly

---

## Potential Counterarguments

| Objection | Response |
|---|---|
| "Rust has a steep learning curve" | Steeper initial curve, but dramatically fewer debugging hours. Net time savings within months. The framework itself reduces what you need to learn. |
| "Our existing codes work fine" | For single-physics, yes. But every coupling project proves otherwise, and maintenance cost grows every year. |
| "Fortran is faster for numerics" | Benchmarks show Rust matches Fortran/C++ performance. LLVM generates equivalent machine code. |
| "Nobody in our group knows Rust" | This is a workforce investment argument, not a technical one. The same was said about Python 15 years ago. |
| "We'd be locked into one framework" | Plugins are plain Rust functions — they can be extracted and used standalone. The framework is an organizational tool, not a cage. |

---

## Example: DEM-CFD Coupling with a Shared Scheduler

With a user-definable `ScheduleSet`, a coupled DEM-CFD simulation's execution order becomes:

```
DEM_InitialIntegration
  -> CFD_ComputeVoidFraction    (reads particle positions + radii)
  -> CFD_SolveNavierStokes      (uses void fraction field)
  -> DEM_ComputeDragForce       (reads fluid velocity at particle locations)
  -> DEM_Force                  (contact forces + drag forces summed)
  -> DEM_FinalIntegration
```

- Data dependencies are explicit via `Res<T>` / `ResMut<T>`
- Subcycling (DEM runs 100x faster than CFD) uses a simple `run_if` condition
- One config file, one restart file, one output pipeline
- No inter-process communication overhead for the coupling itself

---

## What a Shared Scheduler Solves for Coupling

### Data Transfer and Serialization — Eliminated
- Traditional coupling (e.g., CFDEM = LIGGGHTS + OpenFOAM) requires serializing particle data, sending it over MPI inter-communicators or sockets, and deserializing on the other side
- With a shared scheduler, both physics modules are in the same process — data sharing is a `Res<T>` borrow with zero copies and zero overhead
- This alone removes thousands of lines of glue code and a major source of bugs

### Synchronization Logic — Handled by the Scheduler
- Traditional coupled codes require explicit synchronization: "wait for CFD to finish step N before DEM reads the fluid field"
- With a shared scheduler, system ordering via `before()`/`after()` constraints makes this automatic and compiler-checked
- It is structurally impossible to run the drag force computation before the fluid solve — the scheduler enforces the dependency graph

### Subcycling / Multi-Rate Time Integration — Trivial
- DEM typically needs 10-100x smaller timesteps than CFD
- Traditional approach: outer loop runs CFD, inner loop runs DEM N times, with manual bookkeeping of which step you're on
- Shared scheduler approach: attach a `run_if` condition that gates CFD systems to every Nth step — the scheduler handles everything else

### Build System and Dependency Management — Unified
- Traditional coupling: two separate build systems (often CMake + make, or two CMake projects), version-locked dependencies, platform-specific workarounds
- Shared framework: one `Cargo.toml`, one build command, dependencies resolved automatically
- No more "which version of OpenFOAM is compatible with which version of LIGGGHTS" problems

### Configuration and I/O — Shared
- One TOML config file defines both DEM and CFD parameters
- One restart file captures the full coupled state (particle positions + mesh fields)
- One output pipeline writes both particle and field data (VTP, CSV, binary)
- No format conversion between codes, no separate post-processing workflows

### Debugging and Profiling — Single Process
- Traditional coupling: debugging requires attaching to two separate processes, correlating logs, and reproducing timing-dependent bugs
- Shared framework: one process, one debugger session, one stack trace, one profiler output
- Data races between physics modules are caught at compile time by Rust's borrow checker

### Unit Conventions and Coordinate Systems — Shared by Construction
- Traditional coupling often involves unit conversion at the interface (SI vs CGS, different length scales)
- A shared framework means the same unit system and the same coordinate conventions — no conversion bugs

---

## What a Shared Scheduler Does NOT Solve

### Domain Decomposition Mismatch
- DEM uses particle-based spatial decomposition; CFD uses mesh-based decomposition
- Particles and their containing mesh cells may live on different MPI ranks
- This requires communication regardless of framework — the shared scheduler does not eliminate the need for particle-to-cell mapping across ranks
- However, single-rank or shared-memory configurations avoid this entirely, and the framework makes it easier to implement the mapping as a dedicated system

### Physics Implementation
- The coupling physics itself (drag models, void fraction computation, heat transfer correlations) must still be implemented correctly
- A shared scheduler organizes the execution but does not validate the physics
- Choosing the right drag law (Gidaspow vs Di Felice vs Beetstra) is still the scientist's job

### Mesh Data Structures and Solvers
- CFD requires cells, faces, sparse matrices, pressure-velocity coupling algorithms, and linear solvers
- None of this comes from the scheduler — it must be built or imported as a library
- The scheduler provides the execution skeleton; the CFD developer fills in the numerical methods

### Algorithm Complexity
- Turbulence models, multiphase flow formulations, and non-Newtonian rheology are domain-specific challenges
- A shared framework does not make SIMPLE/PISO easier to implement or LES subgrid models less complex
- It only ensures that once implemented, they compose cleanly with other physics

### Performance Optimization of Inner Loops
- The scheduler has negligible overhead (nanoseconds per step for system dispatch)
- The performance-critical work is inside the systems: matrix assembly, sparse solves, particle contact detection
- SIMD vectorization, cache optimization, and GPU offloading are per-system concerns the framework does not address

### Validation of Coupled Results
- A shared scheduler ensures the code runs in the right order, but not that the coupled physics is correct
- Verification (does the code solve the equations right?) and validation (are we solving the right equations?) remain the scientist's responsibility
- The framework makes it easier to build regression tests, but does not generate them

### Legacy Code Integration
- Existing Fortran/C++ codes cannot be dropped into the framework as-is
- FFI bindings are possible but re-introduce many of the coupling problems the framework avoids
- The full benefit requires new code to be written natively in the framework — this is an investment, not a free lunch

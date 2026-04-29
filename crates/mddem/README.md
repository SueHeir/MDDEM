# mddem

The main umbrella crate for **MDDEM** — a plugin-based simulation framework for Discrete Element Method (DEM) and Molecular Dynamics (MD) simulations.

## What It Does

This crate is the entry point to MDDEM. It re-exports all plugin crates and provides convenient plugin groups so you can set up a complete simulation in just a few lines.

The App / Plugin / Scheduler / TOML / multi-stage run substrate lives in the [grass](https://github.com/elizabeth-suehr/grass) workspace (`grass_app`, `grass_scheduler`, `grass_io`, `grass_mpi`, `grass_derive`); MDDEM consumes it as path dependencies and adds the DEM/MD physics on top.

## Plugin Groups

- **`CorePlugins`** — Infrastructure essentials: input/TOML config, MPI/single-process communication, domain decomposition, neighbor lists, run loop, and output
- **`LJDefaultPlugins`** — MD lennard-jones with lattice initialization, Nose–Hoover thermostat, and measurements (pressure, temperature, energy)
- **`GranularDefaultPlugins`** — DEM granular with Hertz–Mindlin contacts, rotational dynamics, and particle insertion

Combine `CorePlugins` with one of the specialized groups, or add individual plugins for finer control.

## Quick Start

```rust
use mddem::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
       .add_plugins(GranularDefaultPlugins);  // or LJDefaultPlugins for MD
    app.start();
}
```

## The Prelude

Use `mddem::prelude::*` to import the most common types: plugin groups, simulation framework (`App`, `Plugin`), core types (`Atom`, `Config`, `RunState`), and all major DEM and MD plugins.

## Crate Organization

**Framework** ([`grass_*`](https://github.com/elizabeth-suehr/grass)): App / Plugin / Scheduler, TOML loading, multi-stage run loop, MPI abstraction, derive macros

**Infrastructure** (`mddem_*`): core simulation types, neighbor lists, output, box deformation, fixes

**DEM** (`dem_*`): granular contacts, bonds, walls, thermal conduction, contact analysis

**MD** (`md_*`): Lennard-Jones, thermostats, lattices, bonds, measurements, polymers

See `lib.rs` for full crate descriptions and the feature flags section for MPI configuration.

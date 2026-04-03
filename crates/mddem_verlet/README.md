# mddem_verlet

Velocity Verlet translational integration for MDDEM simulations.

## Overview

This crate implements the **velocity Verlet algorithm**, a symplectic, time-reversible time-integration scheme for solving Newton's equations of motion in molecular dynamics (MD) and discrete element method (DEM) simulations.

## The Velocity Verlet Algorithm

Velocity Verlet uses a "kick-drift-kick" decomposition split across two phases:

**Initial integration** (before force computation):
```
v(t + Δt/2) = v(t) + (Δt / 2m) · F(t)    // half-step velocity kick
x(t + Δt)   = x(t) + Δt · v(t + Δt/2)     // full-step position drift
```

**Final integration** (after force computation):
```
v(t + Δt) = v(t + Δt/2) + (Δt / 2m) · F(t + Δt)  // completing velocity kick
```

This decomposition is **second-order accurate in Δt**, conserves energy to O(Δt²) per step, and exactly integrates constant-force motion.

## Key Types

- **`VelocityVerletPlugin`**: Registers integration systems. Can run globally (all stages) or restricted to a single `[[run]]` stage.
- **`initial_integration()`**: Performs half-kick and position drift.
- **`final_integration()`**: Completes the velocity kick.

## Usage

```rust
use mddem_verlet::VelocityVerletPlugin;

// All stages (default)
app.add_plugins(VelocityVerletPlugin::new());

// Single stage
app.add_plugins(VelocityVerletPlugin::for_stage("relaxation"));
```

The plugin schedules integration at `ParticleSimScheduleSet::InitialIntegration` (before forces) and `ParticleSimScheduleSet::FinalIntegration` (after forces).

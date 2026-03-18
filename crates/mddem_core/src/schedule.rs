//! Standard Verlet-style schedule phases for particle simulations (MD/DEM).
//!
//! These enums define the execution order for a velocity-Verlet integration loop.
//! Other domains (e.g. CFD) can define their own schedule phases by implementing
//! [`SchedulePhase`] on a custom enum (or using `#[derive(SchedulePhase)]`).

use sim_scheduler::SchedulePhase;

/// Execution phase within each timestep (the run loop).
///
/// Systems are sorted first by their `ScheduleSet` phase, then topologically within
/// each phase using `before`/`after` constraints. The phases execute in the order
/// listed below, mirroring a standard velocity-Verlet integration cycle:
///
/// | Index | Phase                    | Typical use                          |
/// |-------|--------------------------|--------------------------------------|
/// | 0     | `Setup`                  | Per-step bookkeeping                 |
/// | 1–3   | `Pre/Initial/PostInitialIntegration` | First half-step velocity update |
/// | 4–5   | `Pre/Exchange`           | Particle migration (MPI)             |
/// | 6–7   | `Pre/Neighbor`           | Neighbor-list rebuild                |
/// | 8–10  | `Pre/Force/PostForce`    | Force computation                    |
/// | 11–13 | `Pre/Final/PostFinalIntegration` | Second half-step velocity update |
#[derive(Debug, Clone, Copy)]
pub enum ScheduleSet {
    /// Per-step bookkeeping (e.g., incrementing timestep counters).
    Setup,
    /// Runs before the initial (first half-step) integration.
    PreInitialIntegration,
    /// First half-step velocity update (velocity-Verlet first kick).
    InitialIntegration,
    /// Runs after the initial integration (e.g., position updates).
    PostInitialIntegration,
    /// Runs before particle exchange / migration.
    PreExchange,
    /// Particle exchange across MPI ranks or periodic boundaries.
    Exchange,
    /// Runs before neighbor-list construction.
    PreNeighbor,
    /// Neighbor-list build / rebuild.
    Neighbor,
    /// Runs before force computation (e.g., zeroing force arrays).
    PreForce,
    /// Main force computation phase.
    Force,
    /// Runs after force computation (e.g., force modifications, diagnostics).
    PostForce,
    /// Runs before the final (second half-step) integration.
    PreFinalIntegration,
    /// Second half-step velocity update (velocity-Verlet second kick).
    FinalIntegration,
    /// Runs after final integration (e.g., output, end-of-step fixes).
    PostFinalIntegration,
}

/// Execution phase during one-time setup (before the run loop begins).
///
/// Setup systems run once per stage in order: `PreSetup` → `Setup` → `PostSetup`.
#[derive(Debug, Clone, Copy)]
pub enum ScheduleSetupSet {
    /// Runs before the main setup phase (e.g., early resource initialization).
    PreSetup,
    /// Main setup phase (e.g., creating neighbor lists, reading restart files).
    Setup,
    /// Runs after setup (e.g., initial force computation, diagnostics).
    PostSetup,
}

impl SchedulePhase for ScheduleSet {
    fn to_index(&self) -> u32 {
        match self {
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
    fn name(&self) -> &'static str {
        match self {
            ScheduleSet::Setup => "Setup",
            ScheduleSet::PreInitialIntegration => "PreInitialIntegration",
            ScheduleSet::InitialIntegration => "InitialIntegration",
            ScheduleSet::PostInitialIntegration => "PostInitialIntegration",
            ScheduleSet::PreExchange => "PreExchange",
            ScheduleSet::Exchange => "Exchange",
            ScheduleSet::PreNeighbor => "PreNeighbor",
            ScheduleSet::Neighbor => "Neighbor",
            ScheduleSet::PreForce => "PreForce",
            ScheduleSet::Force => "Force",
            ScheduleSet::PostForce => "PostForce",
            ScheduleSet::PreFinalIntegration => "PreFinalIntegration",
            ScheduleSet::FinalIntegration => "FinalIntegration",
            ScheduleSet::PostFinalIntegration => "PostFinalIntegration",
        }
    }
}

/// Returns Verlet-specific schedule warnings for the given phase names.
///
/// This function is registered as the scheduler's `warning_fn` callback by
/// [`CorePlugins`] so that the generic scheduler can emit particle-simulation
/// warnings without knowing about Verlet phase names directly.
pub fn verlet_schedule_warnings(phase_names: &[&str]) -> Vec<String> {
    let mut warnings = Vec::new();

    let is_standard_schedule = phase_names.iter().any(|n| {
        matches!(
            *n,
            "Force"
                | "InitialIntegration"
                | "FinalIntegration"
                | "Setup"
                | "PreInitialIntegration"
                | "PostInitialIntegration"
                | "PreExchange"
                | "Exchange"
                | "PreNeighbor"
                | "Neighbor"
                | "PreForce"
                | "PostForce"
                | "PreFinalIntegration"
                | "PostFinalIntegration"
        )
    });

    if !is_standard_schedule {
        return warnings;
    }

    let has_phase = |name: &str| -> bool { phase_names.iter().any(|n| *n == name) };

    // Count distinct non-Setup phase names
    let update_count: usize = phase_names
        .iter()
        .filter(|n| **n != "Setup")
        .collect::<std::collections::HashSet<_>>()
        .len();

    // 1. No Force systems
    if update_count > 0 && !has_phase("Force") {
        warnings.push(
            "[Schedule Warning] No systems registered in the Force schedule set. \
             Forces will not be computed. Did you forget a force plugin?"
                .to_string(),
        );
    }

    // 2. Asymmetric Verlet
    let has_initial = has_phase("InitialIntegration") || has_phase("PreInitialIntegration");
    let has_final = has_phase("FinalIntegration") || has_phase("PostFinalIntegration");
    if has_initial && !has_final {
        warnings.push(
            "[Schedule Warning] InitialIntegration has systems but FinalIntegration is empty. \
             This produces an asymmetric Verlet integration."
                .to_string(),
        );
    } else if !has_initial && has_final {
        warnings.push(
            "[Schedule Warning] FinalIntegration has systems but InitialIntegration is empty. \
             This produces an asymmetric Verlet integration."
                .to_string(),
        );
    }

    // 3. No integrator
    if update_count > 2 && !has_initial && !has_final {
        warnings.push(
            "[Schedule Warning] Schedule has update systems but no integrator \
             (neither InitialIntegration nor FinalIntegration has systems). \
             Particles will not move."
                .to_string(),
        );
    }

    warnings
}

impl SchedulePhase for ScheduleSetupSet {
    fn to_index(&self) -> u32 {
        match self {
            ScheduleSetupSet::PreSetup => 0,
            ScheduleSetupSet::Setup => 1,
            ScheduleSetupSet::PostSetup => 2,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            ScheduleSetupSet::PreSetup => "PreSetup",
            ScheduleSetupSet::Setup => "Setup",
            ScheduleSetupSet::PostSetup => "PostSetup",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warns_no_force_systems() {
        let phases = vec!["InitialIntegration", "FinalIntegration"];
        let warnings = verlet_schedule_warnings(&phases);
        assert!(
            warnings.iter().any(|w| w.contains("Force")),
            "Expected warning about missing Force systems, got: {:?}",
            warnings
        );
    }

    #[test]
    fn warns_asymmetric_verlet() {
        let phases = vec!["InitialIntegration", "Force"];
        let warnings = verlet_schedule_warnings(&phases);
        assert!(
            warnings.iter().any(|w| w.contains("asymmetric")),
            "Expected warning about asymmetric Verlet, got: {:?}",
            warnings
        );
    }

    #[test]
    fn warns_no_integrator() {
        let phases = vec!["Force", "PostForce", "Exchange"];
        let warnings = verlet_schedule_warnings(&phases);
        assert!(
            warnings.iter().any(|w| w.contains("no integrator")),
            "Expected warning about no integrator, got: {:?}",
            warnings
        );
    }

    #[test]
    fn no_warnings_for_normal_schedule() {
        let phases = vec!["InitialIntegration", "Force", "FinalIntegration"];
        let warnings = verlet_schedule_warnings(&phases);
        assert!(
            warnings.is_empty(),
            "Expected no warnings, got: {:?}",
            warnings
        );
    }

    #[test]
    fn no_warnings_for_custom_schedule() {
        let phases = vec!["PhaseA", "PhaseB", "PhaseC"];
        let warnings = verlet_schedule_warnings(&phases);
        assert!(
            warnings.is_empty(),
            "Expected no warnings for custom schedule, got: {:?}",
            warnings
        );
    }
}

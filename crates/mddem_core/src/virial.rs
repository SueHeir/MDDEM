//! Full 3×3 symmetric virial stress tensor shared across all force types.

use std::any::TypeId;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;

use crate::{RunState, ParticleSimScheduleSet};

/// Symmetric virial stress tensor (upper triangle: xx, yy, zz, xy, xz, yz).
///
/// Accumulated by all force types (LJ, bond, contact) during the Force stage.
/// Zeroed each step at PreForce by [`zero_virial_stress`].
///
/// Sign convention:
/// - `dx, dy, dz` = `pos[j] - pos[i]`
/// - `fx, fy, fz` = force on atom i from atom j
/// - Pressure: `P = NkT/V - trace/(3V)` (repulsion → negative trace → positive P)
pub struct VirialStress {
    pub xx: f64,
    pub yy: f64,
    pub zz: f64,
    pub xy: f64,
    pub xz: f64,
    pub yz: f64,
    /// Whether virial should be computed this step.
    pub active: bool,
    /// `None` = every step, `Some(n)` = every n steps.
    interval: Option<usize>,
}

impl Default for VirialStress {
    fn default() -> Self {
        VirialStress {
            xx: 0.0,
            yy: 0.0,
            zz: 0.0,
            xy: 0.0,
            xz: 0.0,
            yz: 0.0,
            active: true,
            interval: None,
        }
    }
}

impl VirialStress {
    pub fn zero(&mut self) {
        self.xx = 0.0;
        self.yy = 0.0;
        self.zz = 0.0;
        self.xy = 0.0;
        self.xz = 0.0;
        self.yz = 0.0;
    }

    /// Accumulate a pairwise virial contribution.
    ///
    /// `dx, dy, dz` = pos\[j\] - pos\[i\]; `fx, fy, fz` = force on atom i from j.
    #[inline]
    pub fn add_pair(&mut self, dx: f64, dy: f64, dz: f64, fx: f64, fy: f64, fz: f64) {
        self.xx += dx * fx;
        self.yy += dy * fy;
        self.zz += dz * fz;
        self.xy += dx * fy;
        self.xz += dx * fz;
        self.yz += dy * fz;
    }

    /// Trace of the tensor: xx + yy + zz.
    pub fn trace(&self) -> f64 {
        self.xx + self.yy + self.zz
    }

    /// Set the virial computation interval. Takes the minimum of the current
    /// and new interval so all consumers get virial data when they need it.
    pub fn set_interval(&mut self, every: usize) {
        if every == 0 {
            return;
        }
        self.interval = Some(match self.interval {
            None => every,
            Some(cur) => cur.min(every),
        });
    }
}

/// Plugin that registers [`VirialStress`] and the [`zero_virial_stress`] system.
///
/// Safe to add from multiple force plugins — only the first registration takes effect.
pub struct VirialStressPlugin;

impl Plugin for VirialStressPlugin {
    fn is_unique(&self) -> bool {
        false
    }

    fn build(&self, app: &mut App) {
        // Guard: only register once
        if app
            .get_mut_resource(TypeId::of::<VirialStress>())
            .is_some()
        {
            return;
        }
        app.add_resource(VirialStress::default());
        app.add_update_system(zero_virial_stress, ParticleSimScheduleSet::PreForce);
    }
}

/// Zero the virial tensor before force computation each step.
/// Sets `active` based on the interval and current step.
pub fn zero_virial_stress(mut virial: ResMut<VirialStress>, run_state: Res<RunState>) {
    virial.active = match virial.interval {
        None => true,
        Some(n) => run_state.total_cycle.is_multiple_of(n),
    };
    if virial.active {
        virial.zero();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virial_add_pair_trace() {
        let mut v = VirialStress::default();
        // dx=1, dy=0, dz=0, fx=-10, fy=0, fz=0 (repulsive along x)
        v.add_pair(1.0, 0.0, 0.0, -10.0, 0.0, 0.0);
        assert!((v.xx - (-10.0)).abs() < 1e-15);
        assert!((v.trace() - (-10.0)).abs() < 1e-15);
    }

    #[test]
    fn virial_zero() {
        let mut v = VirialStress::default();
        v.add_pair(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        v.zero();
        assert!((v.trace()).abs() < 1e-15);
    }
}

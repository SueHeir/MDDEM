//! MD bond potentials: harmonic springs and FENE (finitely extensible nonlinear elastic).
//!
//! Reads bond topology from [`BondStore`] and applies pairwise bond forces.
//! Supports two bond styles:
//! - **Harmonic**: `U = k/2 (r - r0)^2`
//! - **FENE**: `U = -0.5 k R0^2 ln(1 - (r/R0)^2)` (used in Kremer-Grest bead-spring model)

use mddem_app::prelude::*;
use mddem_core::{Atom, AtomDataRegistry, BondStore, Config, Domain};
use mddem_scheduler::prelude::*;
use serde::Deserialize;

// ── Config ──────────────────────────────────────────────────────────────────

fn default_style() -> String {
    "fene".to_string()
}
fn default_k() -> f64 {
    30.0
}
fn default_r0() -> f64 {
    1.5
}

/// TOML `[md_bond]` configuration.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct MdBondConfig {
    /// Bond style: "harmonic" or "fene".
    #[serde(default = "default_style")]
    pub style: String,
    /// Spring constant (energy / length^2 for harmonic, energy / length^2 for FENE).
    #[serde(default = "default_k")]
    pub k: f64,
    /// Maximum extension for FENE, or equilibrium length for harmonic.
    #[serde(default = "default_r0")]
    pub r0: f64,
}

impl Default for MdBondConfig {
    fn default() -> Self {
        MdBondConfig {
            style: "fene".to_string(),
            k: 30.0,
            r0: 1.5,
        }
    }
}

/// Parsed bond style enum for fast dispatch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BondStyle {
    Harmonic,
    Fene,
}

pub struct MdBondState {
    pub style: BondStyle,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// MD bond force plugin. Requires [`BondPlugin`](mddem_core::BondPlugin) to be registered.
pub struct MdBondPlugin;

impl Plugin for MdBondPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[md_bond]
style = "fene"    # "harmonic" or "fene"
k = 30.0          # spring constant
r0 = 1.5          # max extension (FENE) or equilibrium length (harmonic)"#,
        )
    }

    fn build(&self, app: &mut App) {
        let config = Config::load::<MdBondConfig>(app, "md_bond");

        let style = match config.style.as_str() {
            "harmonic" => BondStyle::Harmonic,
            "fene" => BondStyle::Fene,
            other => panic!("Unknown md_bond style: '{}'. Use 'harmonic' or 'fene'.", other),
        };

        app.add_resource(MdBondState { style });

        // Register BondStore if not already present
        app.add_plugins(mddem_core::BondPlugin);

        app.add_update_system(md_bond_force, ScheduleSet::Force);
    }
}

// ── Force computation ───────────────────────────────────────────────────────

pub fn md_bond_force(
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    bond_config: Res<MdBondConfig>,
    bond_state: Res<MdBondState>,
    domain: Res<Domain>,
) {
    let bond_store = match registry.get::<BondStore>() {
        Some(bs) => bs,
        None => return,
    };

    let nlocal = atoms.nlocal as usize;
    let k = bond_config.k;
    let r0_config = bond_config.r0;
    let style = bond_state.style;

    let lx = domain.size[0];
    let ly = domain.size[1];
    let lz = domain.size[2];
    let half_lx = lx * 0.5;
    let half_ly = ly * 0.5;
    let half_lz = lz * 0.5;

    // Build tag-to-local-index map for partner lookup
    let total = atoms.len();
    let max_tag = atoms.tag[..total].iter().cloned().max().unwrap_or(0) as usize;
    let mut tag_map = vec![usize::MAX; max_tag + 1];
    for i in 0..total {
        let t = atoms.tag[i] as usize;
        if t <= max_tag {
            tag_map[t] = i;
        }
    }

    for i in 0..nlocal {
        for bond in &bond_store.bonds[i] {
            let partner_tag = bond.partner_tag as usize;
            if partner_tag > max_tag {
                continue;
            }
            let j = tag_map[partner_tag];
            if j == usize::MAX {
                continue;
            }

            // Only compute for i < j (by tag) to avoid double-counting
            if atoms.tag[i] >= bond.partner_tag {
                continue;
            }

            let mut dx = atoms.pos[j][0] - atoms.pos[i][0];
            let mut dy = atoms.pos[j][1] - atoms.pos[i][1];
            let mut dz = atoms.pos[j][2] - atoms.pos[i][2];

            // Minimum image convention
            if domain.is_periodic[0] {
                if dx > half_lx { dx -= lx; } else if dx < -half_lx { dx += lx; }
            }
            if domain.is_periodic[1] {
                if dy > half_ly { dy -= ly; } else if dy < -half_ly { dy += ly; }
            }
            if domain.is_periodic[2] {
                if dz > half_lz { dz -= lz; } else if dz < -half_lz { dz += lz; }
            }

            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();

            if r < 1e-20 {
                continue;
            }

            // fpair > 0 means attractive (pulls i toward j along dx)
            let fpair = match style {
                BondStyle::Harmonic => {
                    // F_on_i = k (r - r0) / r along r_hat_ij
                    let r0_bond = bond.r0;
                    k * (r - r0_bond) / r
                }
                BondStyle::Fene => {
                    // FENE: F_on_i = k r / (1 - (r/R0)^2) / r along r_hat_ij
                    // Always attractive (pulls toward r=0)
                    let ratio2 = r2 / (r0_config * r0_config);
                    if ratio2 >= 1.0 {
                        eprintln!(
                            "WARNING: FENE bond exceeded maximum extension! r={:.4}, R0={:.4}, tags=({},{})",
                            r, r0_config, atoms.tag[i], bond.partner_tag
                        );
                        // Apply a large restoring force
                        k * r * 100.0
                    } else {
                        k / (1.0 - ratio2)
                    }
                }
            };

            // Force on atom i toward atom j
            let fx = fpair * dx;
            let fy = fpair * dy;
            let fz = fpair * dz;

            atoms.force[i][0] += fx;
            atoms.force[i][1] += fy;
            atoms.force[i][2] += fz;
            atoms.force[j][0] -= fx;
            atoms.force[j][1] -= fy;
            atoms.force[j][2] -= fz;
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::{AtomDataRegistry, BondEntry, BondStore};

    fn make_bonded_pair_app(distance: f64, style: BondStyle, k: f64, r0: f64) -> App {
        let mut app = App::new();

        let bond_config = MdBondConfig {
            style: match style {
                BondStyle::Harmonic => "harmonic".to_string(),
                BondStyle::Fene => "fene".to_string(),
            },
            k,
            r0,
        };
        app.add_resource(bond_config);
        app.add_resource(MdBondState { style });
        app.add_resource(Domain::default());
        app.add_resource(mddem_core::RunState::default());

        let mut atom = Atom::new();
        atom.push_test_atom(0, [0.0, 0.0, 0.0], 0.5, 1.0);
        atom.push_test_atom(1, [distance, 0.0, 0.0], 0.5, 1.0);
        atom.nlocal = 2;
        atom.natoms = 2;

        // Create bond store and registry
        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry {
            partner_tag: 1,
            bond_type: 0,
            r0,
        }]);
        bond_store.bonds.push(vec![BondEntry {
            partner_tag: 0,
            bond_type: 0,
            r0,
        }]);

        let mut registry = AtomDataRegistry::new();
        registry.register(bond_store);
        app.add_resource(registry);

        app.add_resource(atom);
        app.add_update_system(md_bond_force, ScheduleSet::Force);
        app.organize_systems();
        app
    }

    #[test]
    fn harmonic_restoring_force_stretched() {
        // Bond at r=1.5, r0=1.0 => stretched => attractive force pulling atoms together
        let mut app = make_bonded_pair_app(1.5, BondStyle::Harmonic, 100.0, 1.0);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Atom 0 should be pulled in +x toward atom 1
        assert!(atom.force[0][0] > 0.0, "atom 0 should be pulled +x: got {}", atom.force[0][0]);
        assert!(atom.force[1][0] < 0.0, "atom 1 should be pulled -x: got {}", atom.force[1][0]);
    }

    #[test]
    fn harmonic_restoring_force_compressed() {
        // Bond at r=0.5, r0=1.0 => compressed => repulsive force pushing atoms apart
        let mut app = make_bonded_pair_app(0.5, BondStyle::Harmonic, 100.0, 1.0);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(atom.force[0][0] < 0.0, "atom 0 should be pushed -x: got {}", atom.force[0][0]);
        assert!(atom.force[1][0] > 0.0, "atom 1 should be pushed +x: got {}", atom.force[1][0]);
    }

    #[test]
    fn harmonic_newtons_third_law() {
        let mut app = make_bonded_pair_app(1.3, BondStyle::Harmonic, 50.0, 1.0);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            (atom.force[0][0] + atom.force[1][0]).abs() < 1e-10,
            "Newton's 3rd law violated: {} + {} != 0",
            atom.force[0][0], atom.force[1][0]
        );
    }

    #[test]
    fn harmonic_force_magnitude() {
        // F = k * (r - r0) = 100 * (1.5 - 1.0) = 50.0
        let mut app = make_bonded_pair_app(1.5, BondStyle::Harmonic, 100.0, 1.0);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        let fx = atom.force[0][0];
        assert!((fx - 50.0).abs() < 1e-10, "expected force 50.0, got {}", fx);
    }

    #[test]
    fn fene_attractive_force() {
        // FENE bond at r=0.8, R0=1.5, k=30
        let mut app = make_bonded_pair_app(0.8, BondStyle::Fene, 30.0, 1.5);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        // FENE is always attractive (pulls toward r=0)
        assert!(atom.force[0][0] > 0.0, "FENE should pull atom 0 toward atom 1: got {}", atom.force[0][0]);
        assert!(atom.force[1][0] < 0.0, "FENE should pull atom 1 toward atom 0: got {}", atom.force[1][0]);
    }

    #[test]
    fn fene_newtons_third_law() {
        let mut app = make_bonded_pair_app(1.0, BondStyle::Fene, 30.0, 1.5);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!(
            (atom.force[0][0] + atom.force[1][0]).abs() < 1e-10,
            "Newton's 3rd law violated for FENE"
        );
    }

    #[test]
    fn fene_diverges_near_r0() {
        // At r close to R0, force should be very large
        let mut app = make_bonded_pair_app(1.49, BondStyle::Fene, 30.0, 1.5);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        let fx = atom.force[0][0].abs();
        assert!(fx > 100.0, "FENE force should be large near R0, got {}", fx);
    }
}

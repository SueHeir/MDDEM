//! MD bond potentials: harmonic springs, FENE, and bond angle (cosine bending).
//!
//! Reads bond topology from [`BondStore`] and applies pairwise bond forces.
//! Optionally reads angle topology from [`AngleStore`] for 3-body bending forces.
//! Supports two bond styles:
//! - **Harmonic**: `U = k/2 (r - r0)^2`
//! - **FENE**: `U = -0.5 k R0^2 ln(1 - (r/R0)^2)` (used in Kremer-Grest bead-spring model)
//!
//! Bond angle potential (cosine bending):
//! - `U(θ) = k_angle (1 - cos θ)` where θ is the angle between consecutive bonds

use mddem_app::prelude::*;
use mddem_core::{AnglePlugin, AngleStore, Atom, AtomDataRegistry, BondStore, Config, Domain, VirialStress};
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
fn default_k_angle() -> f64 {
    0.0
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
    /// Bond angle bending stiffness. U(θ) = k_angle (1 - cos θ).
    /// Set to 0.0 (default) for fully flexible chains.
    #[serde(default = "default_k_angle")]
    pub k_angle: f64,
}

impl Default for MdBondConfig {
    fn default() -> Self {
        MdBondConfig {
            style: "fene".to_string(),
            k: 30.0,
            r0: 1.5,
            k_angle: 0.0,
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
r0 = 1.5          # max extension (FENE) or equilibrium length (harmonic)
k_angle = 0.0     # bond angle stiffness (0 = fully flexible)"#,
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

        // Register AngleStore if angle potential is active
        if config.k_angle > 0.0 {
            app.add_plugins(AnglePlugin);
        }

        // Register VirialStressPlugin so bond forces contribute to pressure
        app.add_plugins(mddem_core::VirialStressPlugin);

        app.add_update_system(md_bond_force, ScheduleSet::Force);
    }
}

// ── Force computation ───────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn md_bond_force(
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    bond_config: Res<MdBondConfig>,
    bond_state: Res<MdBondState>,
    domain: Res<Domain>,
    virial: Option<ResMut<VirialStress>>,
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

    let virial_active = virial.as_ref().map_or(false, |v| v.active);
    let mut vxx = 0.0f64;
    let mut vyy = 0.0f64;
    let mut vzz = 0.0f64;
    let mut vxy = 0.0f64;
    let mut vxz = 0.0f64;
    let mut vyz = 0.0f64;

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
                    // FENE potential: U = -0.5 k R0^2 ln(1 - (r/R0)^2)
                    // Force magnitude: F = -dU/dr = k r / (1 - (r/R0)^2)
                    //
                    // `fpair` is F/r (force magnitude divided by distance) because
                    // it gets multiplied by the unnormalized displacement vector
                    // (dx, dy, dz) to produce force components:
                    //   fx = fpair * dx = (F/r) * dx = F * (dx/r)
                    // This avoids a separate normalization step.
                    let ratio2 = r2 / (r0_config * r0_config);
                    if ratio2 >= 1.0 {
                        eprintln!(
                            "WARNING: FENE bond exceeded maximum extension! r={:.4}, R0={:.4}, tags=({},{})",
                            r, r0_config, atoms.tag[i], bond.partner_tag
                        );
                        // Cap at the force value at r = 0.99 * R0 as an emergency restoring force.
                        // F(0.99*R0)/r ≈ k / (1 - 0.99^2) = k / 0.0199 ≈ 50.25 * k
                        let cap_ratio2 = 0.99 * 0.99;
                        k / (1.0 - cap_ratio2)
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

            // Accumulate virial: W_ij = r_ij ⊗ f_ij
            if virial_active {
                vxx += dx * fx;
                vyy += dy * fy;
                vzz += dz * fz;
                vxy += dx * fy;
                vxz += dx * fz;
                vyz += dy * fz;
            }
        }
    }

    // Write accumulated virial
    if virial_active {
        if let Some(mut virial) = virial {
            virial.xx += vxx;
            virial.yy += vyy;
            virial.zz += vzz;
            virial.xy += vxy;
            virial.xz += vxz;
            virial.yz += vyz;
        }
    }

    // Bond angle forces
    if bond_config.k_angle > 0.0 {
        if let Some(angle_store) = registry.get::<AngleStore>() {
            compute_angle_forces(
                &mut atoms,
                &angle_store,
                &tag_map,
                max_tag,
                bond_config.k_angle,
                &domain,
            );
        }
    }
}

// ── Angle force computation ─────────────────────────────────────────────────

/// Compute bond angle forces for cosine bending potential: U(θ) = k(1 - cos θ).
///
/// For three atoms i—j—k with j as the central atom:
/// - θ is the angle at j between bonds j→i and j→k
/// - Forces are applied to all three atoms to maintain Newton's 3rd law
fn compute_angle_forces(
    atoms: &mut Atom,
    angle_store: &AngleStore,
    tag_map: &[usize],
    max_tag: usize,
    k_angle: f64,
    domain: &Domain,
) {
    let nlocal = atoms.nlocal as usize;
    let lx = domain.size[0];
    let ly = domain.size[1];
    let lz = domain.size[2];
    let half_lx = lx * 0.5;
    let half_ly = ly * 0.5;
    let half_lz = lz * 0.5;

    for j_local in 0..nlocal {
        if j_local >= angle_store.angles.len() {
            continue;
        }

        for angle in &angle_store.angles[j_local] {
            let ti = angle.tag_i as usize;
            let tk = angle.tag_k as usize;
            if ti > max_tag || tk > max_tag {
                continue;
            }
            let i = tag_map[ti];
            let k = tag_map[tk];
            if i == usize::MAX || k == usize::MAX {
                continue;
            }

            // Vector from j to i
            let mut d_ji = [
                atoms.pos[i][0] - atoms.pos[j_local][0],
                atoms.pos[i][1] - atoms.pos[j_local][1],
                atoms.pos[i][2] - atoms.pos[j_local][2],
            ];
            // Vector from j to k
            let mut d_jk = [
                atoms.pos[k][0] - atoms.pos[j_local][0],
                atoms.pos[k][1] - atoms.pos[j_local][1],
                atoms.pos[k][2] - atoms.pos[j_local][2],
            ];

            // Minimum image
            if domain.is_periodic[0] {
                if d_ji[0] > half_lx { d_ji[0] -= lx; } else if d_ji[0] < -half_lx { d_ji[0] += lx; }
                if d_jk[0] > half_lx { d_jk[0] -= lx; } else if d_jk[0] < -half_lx { d_jk[0] += lx; }
            }
            if domain.is_periodic[1] {
                if d_ji[1] > half_ly { d_ji[1] -= ly; } else if d_ji[1] < -half_ly { d_ji[1] += ly; }
                if d_jk[1] > half_ly { d_jk[1] -= ly; } else if d_jk[1] < -half_ly { d_jk[1] += ly; }
            }
            if domain.is_periodic[2] {
                if d_ji[2] > half_lz { d_ji[2] -= lz; } else if d_ji[2] < -half_lz { d_ji[2] += lz; }
                if d_jk[2] > half_lz { d_jk[2] -= lz; } else if d_jk[2] < -half_lz { d_jk[2] += lz; }
            }

            let r_ji2 = d_ji[0] * d_ji[0] + d_ji[1] * d_ji[1] + d_ji[2] * d_ji[2];
            let r_jk2 = d_jk[0] * d_jk[0] + d_jk[1] * d_jk[1] + d_jk[2] * d_jk[2];

            if r_ji2 < 1e-20 || r_jk2 < 1e-20 {
                continue;
            }

            let r_ji = r_ji2.sqrt();
            let r_jk = r_jk2.sqrt();

            // cos(θ) = (d_ji · d_jk) / (|d_ji| |d_jk|)
            let dot = d_ji[0] * d_jk[0] + d_ji[1] * d_jk[1] + d_ji[2] * d_jk[2];
            let cos_theta = (dot / (r_ji * r_jk)).clamp(-1.0, 1.0);

            // U(θ) = k(1 - cos θ)
            // dU/dθ = k sin θ
            // For the force on atom i:
            //   f_i = -k_angle * d(cos θ)/d(r_i)
            //       = -k_angle * (d_jk/(r_ji*r_jk) - cos_theta * d_ji/r_ji^2)
            // Similarly for atom k.

            let inv_r_ji = 1.0 / r_ji;
            let inv_r_jk = 1.0 / r_jk;

            // Force on atom i: F_i = -k_angle * (r_jk_hat/r_ji - cos_theta * r_ji_hat/r_ji)
            //                      = -k_angle / r_ji * (r_jk_hat - cos_theta * r_ji_hat)
            // where r_ji_hat = d_ji/r_ji, r_jk_hat = d_jk/r_jk
            let mut f_i = [0.0f64; 3];
            let mut f_k = [0.0f64; 3];

            // F_i = -dU/dr_i = k * d(cos θ)/dr_i
            // d(cos θ)/dr_i = (d_jk/(r_ji*r_jk) - cos_theta * d_ji/r_ji^2)
            for d in 0..3 {
                f_i[d] = k_angle * inv_r_ji * (d_jk[d] * inv_r_jk - cos_theta * d_ji[d] * inv_r_ji);
                f_k[d] = k_angle * inv_r_jk * (d_ji[d] * inv_r_ji - cos_theta * d_jk[d] * inv_r_jk);
            }

            // Force on central atom j = -(F_i + F_k) to conserve momentum
            atoms.force[i][0] += f_i[0];
            atoms.force[i][1] += f_i[1];
            atoms.force[i][2] += f_i[2];
            atoms.force[k][0] += f_k[0];
            atoms.force[k][1] += f_k[1];
            atoms.force[k][2] += f_k[2];
            atoms.force[j_local][0] -= f_i[0] + f_k[0];
            atoms.force[j_local][1] -= f_i[1] + f_k[1];
            atoms.force[j_local][2] -= f_i[2] + f_k[2];
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
            k_angle: 0.0,
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

    // ── FENE divergence scaling ────────────────────────────────────────────

    #[test]
    fn fene_force_scaling_as_r_approaches_r0() {
        // FENE force: F = k * r / (1 - (r/R0)^2)
        // As r → R0, the denominator → 0, so F → ∞.
        // Verify that force at r = R0 - ε is approximately k*R0 / (2ε/R0)
        // i.e. F ≈ k*R0^2 / (2ε) for small ε.
        let k = 30.0;
        let r0 = 1.5;

        let eps1 = 0.01;
        let eps2 = 0.005; // half of eps1

        let mut app1 = make_bonded_pair_app(r0 - eps1, BondStyle::Fene, k, r0);
        app1.run();
        let f1 = app1.get_resource_ref::<Atom>().unwrap().force[0][0].abs();

        let mut app2 = make_bonded_pair_app(r0 - eps2, BondStyle::Fene, k, r0);
        app2.run();
        let f2 = app2.get_resource_ref::<Atom>().unwrap().force[0][0].abs();

        // When ε halves, force should approximately double (1/ε scaling)
        let ratio = f2 / f1;
        assert!(
            (ratio - 2.0).abs() < 0.3,
            "FENE force should scale as ~1/ε near R0: f2/f1={:.3}, expected ~2.0",
            ratio
        );
    }

    #[test]
    fn fene_exact_force_value() {
        // Verify exact FENE force at a specific distance.
        // F_on_i = k * r / (1 - (r/R0)^2) directed toward j (in +x).
        // At r=1.0, R0=1.5, k=30:
        //   ratio2 = 1.0 / 2.25 = 0.4444...
        //   fpair = k / (1 - ratio2) = 30 / 0.5556 = 54.0
        //   F = fpair * dx = 54.0 * 1.0 = 54.0
        let k = 30.0;
        let r0 = 1.5;
        let r = 1.0;
        let expected_fpair = k / (1.0 - (r * r) / (r0 * r0));
        let expected_force = expected_fpair * r; // F = fpair * dx, dx = r along x

        let mut app = make_bonded_pair_app(r, BondStyle::Fene, k, r0);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        let fx = atom.force[0][0];
        assert!(
            (fx - expected_force).abs() < 1e-10,
            "FENE force mismatch: got {}, expected {}",
            fx,
            expected_force
        );
    }

    // ── FENE energy conservation (elastic two-body oscillation) ────────────

    /// Manually integrate a FENE bonded pair using velocity Verlet,
    /// computing potential energy analytically at each step.
    /// U_FENE = -0.5 * k * R0^2 * ln(1 - (r/R0)^2)
    #[test]
    fn fene_energy_conservation_two_body() {
        let k = 30.0;
        let r0_max = 1.5;
        let mass = 1.0;
        let dt = 0.0001; // small timestep for accurate conservation
        let nsteps = 10_000;

        // Initial condition: two particles, separation r=0.8 (< R0), at rest.
        // Reduced mass μ = m/2 for equal masses.
        let mut x1 = 0.0_f64;
        let mut x2 = 0.8_f64;
        let mut v1 = 0.0_f64;
        let mut v2 = 0.0_f64;

        let fene_potential = |r: f64| -> f64 {
            let ratio2 = (r / r0_max).powi(2);
            -0.5 * k * r0_max * r0_max * (1.0 - ratio2).ln()
        };
        let fene_force_on_1 = |x1: f64, x2: f64| -> f64 {
            // Force on particle 1 due to FENE bond to particle 2
            let dx = x2 - x1;
            let r = dx.abs();
            let ratio2 = (r / r0_max).powi(2);
            let fpair = k / (1.0 - ratio2);
            fpair * dx // attractive: pulls 1 toward 2
        };

        let r_init = (x2 - x1).abs();
        let pe_init = fene_potential(r_init);
        let ke_init = 0.5 * mass * (v1 * v1 + v2 * v2);
        let e_total_init = ke_init + pe_init;

        let mut max_energy_error = 0.0_f64;

        for _ in 0..nsteps {
            // Velocity Verlet
            let f1 = fene_force_on_1(x1, x2);
            let f2 = -f1; // Newton's 3rd law

            // Half-kick
            v1 += 0.5 * dt * f1 / mass;
            v2 += 0.5 * dt * f2 / mass;

            // Drift
            x1 += v1 * dt;
            x2 += v2 * dt;

            // New forces
            let f1_new = fene_force_on_1(x1, x2);
            let f2_new = -f1_new;

            // Half-kick
            v1 += 0.5 * dt * f1_new / mass;
            v2 += 0.5 * dt * f2_new / mass;

            // Check energy
            let r = (x2 - x1).abs();
            let pe = fene_potential(r);
            let ke = 0.5 * mass * (v1 * v1 + v2 * v2);
            let e_total = ke + pe;
            let err = (e_total - e_total_init).abs();
            if err > max_energy_error {
                max_energy_error = err;
            }
        }

        // With dt=0.0001, Velocity Verlet should conserve energy to ~O(dt^2) per step
        assert!(
            max_energy_error < 1e-4,
            "FENE energy drift too large: max error = {:.2e} (expected < 1e-4)",
            max_energy_error
        );
    }

    // ── Harmonic limit: oscillation frequency ──────────────────────────────

    #[test]
    fn fene_harmonic_limit_oscillation_frequency() {
        // For small displacements around r=0, FENE reduces to harmonic:
        //   U ≈ 0.5 * k * r^2  (for r << R0)
        // Two equal-mass particles on a FENE bond have reduced mass μ = m/2
        // and oscillation frequency ω = sqrt(k/μ) = sqrt(2k/m).
        // Period T = 2π/ω.
        //
        // Start from small displacement, measure half-period (time to return
        // to initial separation after one full oscillation).
        let k = 30.0;
        let r0_max = 1.5;
        let mass = 1.0;
        let dt = 0.0001;

        let omega = (2.0_f64 * k / mass).sqrt(); // ω = sqrt(2k/m) for two-body
        let period = 2.0 * std::f64::consts::PI / omega;

        // Start with small displacement (r << R0 for harmonic regime)
        let r_init = 0.05;
        let mut x1 = 0.0_f64;
        let mut x2 = r_init;
        let mut v1 = 0.0_f64;
        let mut v2 = 0.0_f64;

        let fene_force_on_1 = |x1: f64, x2: f64| -> f64 {
            let dx = x2 - x1;
            let r2 = dx * dx;
            let ratio2 = r2 / (r0_max * r0_max);
            let fpair = k / (1.0 - ratio2);
            fpair * dx
        };

        // Integrate for one full period and check that r returns close to r_init
        let total_steps = (period / dt).round() as usize;
        for _ in 0..total_steps {
            let f1 = fene_force_on_1(x1, x2);
            v1 += 0.5 * dt * f1 / mass;
            v2 -= 0.5 * dt * f1 / mass;
            x1 += v1 * dt;
            x2 += v2 * dt;
            let f1_new = fene_force_on_1(x1, x2);
            v1 += 0.5 * dt * f1_new / mass;
            v2 -= 0.5 * dt * f1_new / mass;
        }

        let r_final = x2 - x1;
        let rel_error = ((r_final - r_init) / r_init).abs();
        assert!(
            rel_error < 0.01,
            "After one period, r should return to r_init: r_final={:.6}, r_init={:.6}, rel_error={:.4}",
            r_final, r_init, rel_error
        );
    }

    // ── Harmonic energy conservation ───────────────────────────────────────

    #[test]
    fn harmonic_energy_conservation_two_body() {
        // U_harmonic = 0.5 * k * (r - r0)^2
        // Two equal-mass particles with Velocity Verlet should conserve total energy.
        let k = 50.0;
        let r0_eq = 1.0;
        let mass = 1.0;
        let dt = 0.0001;
        let nsteps = 10_000;

        // Start stretched: r = 1.3
        let mut x1 = 0.0_f64;
        let mut x2 = 1.3_f64;
        let mut v1 = 0.0_f64;
        let mut v2 = 0.0_f64;

        let harmonic_pe = |x1: f64, x2: f64| -> f64 {
            let r = (x2 - x1).abs();
            0.5 * k * (r - r0_eq).powi(2)
        };
        let harmonic_force_on_1 = |x1: f64, x2: f64| -> f64 {
            let dx = x2 - x1;
            let r = dx.abs();
            k * (r - r0_eq) * dx / r // attractive when stretched
        };

        let e_init = 0.5 * mass * (v1 * v1 + v2 * v2) + harmonic_pe(x1, x2);
        let mut max_err = 0.0_f64;

        for _ in 0..nsteps {
            let f1 = harmonic_force_on_1(x1, x2);
            v1 += 0.5 * dt * f1 / mass;
            v2 -= 0.5 * dt * f1 / mass;
            x1 += v1 * dt;
            x2 += v2 * dt;
            let f1_new = harmonic_force_on_1(x1, x2);
            v1 += 0.5 * dt * f1_new / mass;
            v2 -= 0.5 * dt * f1_new / mass;

            let e = 0.5 * mass * (v1 * v1 + v2 * v2) + harmonic_pe(x1, x2);
            let err = (e - e_init).abs();
            if err > max_err {
                max_err = err;
            }
        }

        assert!(
            max_err < 1e-5,
            "Harmonic energy drift: max error = {:.2e}",
            max_err
        );
    }

    // ── FENE force at r=0 is zero ──────────────────────────────────────────

    #[test]
    fn fene_force_zero_at_zero_separation() {
        // FENE: F = k * r / (1 - (r/R0)^2). At r=0, F=0 (equilibrium).
        // We can't set r=0 exactly (skipped for r < 1e-20), but very small r
        // should give very small force.
        let mut app = make_bonded_pair_app(1e-10, BondStyle::Fene, 30.0, 1.5);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        let fx = atom.force[0][0].abs();
        assert!(
            fx < 1e-5,
            "FENE force should be ~0 at r≈0, got {}",
            fx
        );
    }

    // ── FENE cap behavior beyond R0 ────────────────────────────────────────

    #[test]
    fn fene_beyond_r0_caps_force() {
        // When r >= R0, force should be capped (not NaN or infinite)
        let r0 = 1.5;
        let mut app = make_bonded_pair_app(1.6, BondStyle::Fene, 30.0, r0);
        app.run();
        let atom = app.get_resource_ref::<Atom>().unwrap();
        let fx = atom.force[0][0];
        assert!(fx.is_finite(), "FENE force beyond R0 should be finite, got {}", fx);
        assert!(fx > 0.0, "FENE force beyond R0 should be attractive (capped), got {}", fx);
    #[test]
    fn angle_force_straight_chain() {
        // Three atoms in a straight line: θ = π, cos θ = -1
        // U = k(1 - cos θ) = k(1 - (-1)) = 2k => maximum energy, should push toward bend
        use mddem_core::AngleEntry;

        let mut app = App::new();
        let bond_config = MdBondConfig {
            style: "harmonic".to_string(),
            k: 0.0, // no bond forces, test angle only
            r0: 1.0,
            k_angle: 10.0,
        };
        app.add_resource(bond_config);
        app.add_resource(MdBondState { style: BondStyle::Harmonic });
        app.add_resource(Domain::default());
        app.add_resource(mddem_core::RunState::default());

        let mut atom = Atom::new();
        atom.push_test_atom(0, [0.0, 0.0, 0.0], 0.5, 1.0);
        atom.push_test_atom(1, [1.0, 0.0, 0.0], 0.5, 1.0);
        atom.push_test_atom(2, [2.0, 0.0, 0.0], 0.5, 1.0);
        atom.nlocal = 3;
        atom.natoms = 3;

        // Bond store (needed to not crash, but k=0 means no bond forces)
        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry { partner_tag: 1, bond_type: 0, r0: 1.0 }]);
        bond_store.bonds.push(vec![
            BondEntry { partner_tag: 0, bond_type: 0, r0: 1.0 },
            BondEntry { partner_tag: 2, bond_type: 0, r0: 1.0 },
        ]);
        bond_store.bonds.push(vec![BondEntry { partner_tag: 1, bond_type: 0, r0: 1.0 }]);

        // Angle store: central atom 1, endpoints 0 and 2
        let mut angle_store = AngleStore::new();
        angle_store.angles.push(Vec::new()); // atom 0: no angles
        angle_store.angles.push(vec![AngleEntry { tag_i: 0, tag_k: 2, angle_type: 0 }]);
        angle_store.angles.push(Vec::new()); // atom 2: no angles

        let mut registry = AtomDataRegistry::new();
        registry.register(bond_store);
        registry.register(angle_store);
        app.add_resource(registry);
        app.add_resource(atom);

        app.add_update_system(md_bond_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // For a straight line (θ=π), the angle derivative w.r.t. perpendicular
        // displacement is zero along the line but non-zero perpendicular.
        // The gradient of cos(θ) at θ=π is degenerate along the line.
        // Forces should sum to zero (momentum conservation).
        let total_fx = atom.force[0][0] + atom.force[1][0] + atom.force[2][0];
        let total_fy = atom.force[0][1] + atom.force[1][1] + atom.force[2][1];
        let total_fz = atom.force[0][2] + atom.force[1][2] + atom.force[2][2];
        assert!(
            total_fx.abs() < 1e-10 && total_fy.abs() < 1e-10 && total_fz.abs() < 1e-10,
            "Total force should be zero: ({}, {}, {})",
            total_fx, total_fy, total_fz
        );
    }

    #[test]
    fn angle_force_right_angle() {
        // Three atoms at 90°: atom 0 at (0,1,0), atom 1 at origin, atom 2 at (1,0,0)
        // θ = π/2, cos θ = 0, force should try to increase cos θ (straighten chain)
        use mddem_core::AngleEntry;

        let mut app = App::new();
        let bond_config = MdBondConfig {
            style: "harmonic".to_string(),
            k: 0.0,
            r0: 1.0,
            k_angle: 10.0,
        };
        app.add_resource(bond_config);
        app.add_resource(MdBondState { style: BondStyle::Harmonic });
        app.add_resource(Domain::default());
        app.add_resource(mddem_core::RunState::default());

        let mut atom = Atom::new();
        atom.push_test_atom(0, [0.0, 1.0, 0.0], 0.5, 1.0);
        atom.push_test_atom(1, [0.0, 0.0, 0.0], 0.5, 1.0);
        atom.push_test_atom(2, [1.0, 0.0, 0.0], 0.5, 1.0);
        atom.nlocal = 3;
        atom.natoms = 3;

        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry { partner_tag: 1, bond_type: 0, r0: 1.0 }]);
        bond_store.bonds.push(vec![
            BondEntry { partner_tag: 0, bond_type: 0, r0: 1.0 },
            BondEntry { partner_tag: 2, bond_type: 0, r0: 1.0 },
        ]);
        bond_store.bonds.push(vec![BondEntry { partner_tag: 1, bond_type: 0, r0: 1.0 }]);

        let mut angle_store = AngleStore::new();
        angle_store.angles.push(Vec::new());
        angle_store.angles.push(vec![AngleEntry { tag_i: 0, tag_k: 2, angle_type: 0 }]);
        angle_store.angles.push(Vec::new());

        let mut registry = AtomDataRegistry::new();
        registry.register(bond_store);
        registry.register(angle_store);
        app.add_resource(registry);
        app.add_resource(atom);

        app.add_update_system(md_bond_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();

        // Momentum conservation
        let total_fx = atom.force[0][0] + atom.force[1][0] + atom.force[2][0];
        let total_fy = atom.force[0][1] + atom.force[1][1] + atom.force[2][1];
        let total_fz = atom.force[0][2] + atom.force[1][2] + atom.force[2][2];
        assert!(
            total_fx.abs() < 1e-10 && total_fy.abs() < 1e-10 && total_fz.abs() < 1e-10,
            "Total force should be zero: ({}, {}, {})",
            total_fx, total_fy, total_fz
        );

        // At θ=90°, the potential wants to straighten (increase cos θ toward 1).
        // Atom 0 at (0,1,0) should be pushed toward x direction (toward atom 2)
        // Atom 2 at (1,0,0) should be pushed toward y direction (toward atom 0)
        assert!(atom.force[0][0] > 0.0, "atom 0 should be pushed in +x: got {}", atom.force[0][0]);
        assert!(atom.force[2][1] > 0.0, "atom 2 should be pushed in +y: got {}", atom.force[2][1]);
    }

    #[test]
    fn bond_virial_accumulated() {
        // Test that bond forces contribute to virial stress
        let mut app = App::new();

        let bond_config = MdBondConfig {
            style: "harmonic".to_string(),
            k: 100.0,
            r0: 1.0,
            k_angle: 0.0,
        };
        app.add_resource(bond_config);
        app.add_resource(MdBondState { style: BondStyle::Harmonic });
        app.add_resource(Domain::default());
        app.add_resource(mddem_core::RunState::default());
        app.add_resource(VirialStress::default());

        let mut atom = Atom::new();
        atom.push_test_atom(0, [0.0, 0.0, 0.0], 0.5, 1.0);
        atom.push_test_atom(1, [1.5, 0.0, 0.0], 0.5, 1.0);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut bond_store = BondStore::new();
        bond_store.bonds.push(vec![BondEntry { partner_tag: 1, bond_type: 0, r0: 1.0 }]);
        bond_store.bonds.push(vec![BondEntry { partner_tag: 0, bond_type: 0, r0: 1.0 }]);

        let mut registry = AtomDataRegistry::new();
        registry.register(bond_store);
        app.add_resource(registry);
        app.add_resource(atom);

        app.add_update_system(md_bond_force, ScheduleSet::Force);
        app.organize_systems();
        app.run();

        let virial = app.get_resource_ref::<VirialStress>().unwrap();
        // For stretched bond along x: dx=1.5, fx=50 (attractive), so W_xx = 1.5*50 = 75
        assert!(
            (virial.xx - 75.0).abs() < 1e-10,
            "virial xx should be 75.0, got {}",
            virial.xx
        );
        assert!(virial.yy.abs() < 1e-10, "virial yy should be 0");
        assert!(virial.zz.abs() < 1e-10, "virial zz should be 0");
    }
}

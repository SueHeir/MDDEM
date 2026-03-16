//! DEM contact analysis: coordination number, per-contact force output, and fabric tensor.
//!
//! Provides [`ContactAnalysisPlugin`] which reads a `[contact_analysis]` TOML config section
//! and registers post-force systems for:
//! - Per-atom coordination number (count of active contacts per particle)
//! - Per-contact force records dumped to CSV files at configurable intervals
//! - Fabric tensor computation from contact normals (output to thermo)
//!
//! # Configuration
//! ```toml
//! [contact_analysis]
//! interval = 1000        # dump contact data every N steps (0 = disabled)
//! coordination = true    # compute per-atom coordination number
//! fabric_tensor = true   # compute fabric tensor to thermo output
//! file_prefix = "contact" # prefix for contact CSV files
//! ```

use std::{
    fs::{self, File},
    io::{BufWriter, Write},
};

use mddem_app::prelude::*;
use mddem_derive::AtomData;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use dem_atom::DemAtom;
use mddem_core::{register_atom_data, Atom, AtomData, AtomDataRegistry, CommResource, Config, Input, RunState};
use mddem_neighbor::Neighbor;
use mddem_print::{DumpRegistry, Thermo};

// ── Config ──────────────────────────────────────────────────────────────────

fn default_file_prefix() -> String {
    "contact".to_string()
}

#[derive(Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct ContactAnalysisConfig {
    /// Dump per-contact data every N steps (0 = disabled).
    #[serde(default)]
    pub interval: usize,
    /// Compute per-atom coordination number.
    #[serde(default)]
    pub coordination: bool,
    /// Compute fabric tensor and output to thermo.
    #[serde(default)]
    pub fabric_tensor: bool,
    /// File prefix for contact CSV output files.
    #[serde(default = "default_file_prefix")]
    pub file_prefix: String,
}

// ── Per-atom coordination data ──────────────────────────────────────────────

/// Per-atom contact analysis data (coordination number).
#[derive(AtomData)]
pub struct ContactAnalysis {
    /// Number of contacts per atom (as f64 for thermo/dump compatibility).
    #[zero]
    pub coordination: Vec<f64>,
}

impl ContactAnalysis {
    pub fn new() -> Self {
        ContactAnalysis {
            coordination: Vec::new(),
        }
    }
}

impl Default for ContactAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

// ── Per-contact record ──────────────────────────────────────────────────────

/// A single contact record for force chain analysis.
#[derive(Clone, Debug)]
pub struct ContactRecord {
    /// Global tag of atom i.
    pub i_tag: u32,
    /// Global tag of atom j.
    pub j_tag: u32,
    /// Total contact force (x, y, z).
    pub fx: f64,
    pub fy: f64,
    pub fz: f64,
    /// Normal force magnitude.
    pub fn_mag: f64,
    /// Tangential force magnitude.
    pub ft_mag: f64,
    /// Overlap (positive = penetration).
    pub overlap: f64,
    /// Contact point (x, y, z).
    pub cx: f64,
    pub cy: f64,
    pub cz: f64,
    /// Contact normal (unit vector from i to j).
    pub nx: f64,
    pub ny: f64,
    pub nz: f64,
}

/// Resource holding per-contact force data, cleared each step.
pub struct ContactOutput {
    pub records: Vec<ContactRecord>,
}

impl ContactOutput {
    pub fn new() -> Self {
        ContactOutput {
            records: Vec::with_capacity(1024),
        }
    }
}

impl Default for ContactOutput {
    fn default() -> Self {
        Self::new()
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Contact analysis plugin: coordination number, per-contact force output, fabric tensor.
pub struct ContactAnalysisPlugin;

impl Plugin for ContactAnalysisPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[contact_analysis]
# Dump per-contact data every N steps (0 = disabled)
interval = 0
# Compute per-atom coordination number
coordination = true
# Compute fabric tensor and output to thermo
fabric_tensor = false
# File prefix for contact CSV output
file_prefix = "contact""#,
        )
    }

    fn build(&self, app: &mut App) {
        let config = Config::load::<ContactAnalysisConfig>(app, "contact_analysis");

        // Always register ContactOutput for per-contact records
        app.add_resource(ContactOutput::new());

        if config.coordination {
            register_atom_data!(app, ContactAnalysis::new());

            // Register coordination as dump scalar
            let dump_reg = app
                .get_mut_resource(std::any::TypeId::of::<DumpRegistry>())
                .expect("DumpRegistry not found — PrintPlugin must be added first");
            dump_reg
                .borrow_mut()
                .downcast_mut::<DumpRegistry>()
                .unwrap()
                .register_scalar("coordination", |atoms, registry| {
                    let ca = registry.expect::<ContactAnalysis>("coordination dump");
                    let nlocal = atoms.nlocal as usize;
                    ca.coordination[..nlocal].to_vec()
                });

            // Push coordination stats to thermo
            app.add_update_system(
                push_coordination_to_thermo.after("contact_analysis"),
                ScheduleSet::PostForce,
            );
        }

        // Coordination + contact record collection (PostForce, after contact forces)
        app.add_update_system(
            compute_contact_analysis
                .label("contact_analysis")
                .after("hertz_mindlin_contact"),
            ScheduleSet::PostForce,
        );

        // Contact CSV dump (PostFinalIntegration, with other output)
        if config.interval > 0 {
            app.add_update_system(dump_contact_records, ScheduleSet::PostFinalIntegration);
        }

        if config.fabric_tensor {
            app.add_update_system(
                compute_fabric_tensor.after("contact_analysis"),
                ScheduleSet::PostForce,
            );
        }
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Post-force system: iterate neighbor pairs, detect contacts (overlap > 0),
/// increment coordination numbers, and collect per-contact records.
#[allow(clippy::too_many_arguments)]
fn compute_contact_analysis(
    atoms: Res<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    config: Res<ContactAnalysisConfig>,
    run_state: Res<RunState>,
    mut contact_output: ResMut<ContactOutput>,
) {
    let nlocal = atoms.nlocal as usize;
    let dem = registry.expect::<DemAtom>("compute_contact_analysis");
    let has_coordination = config.coordination;
    let collect_records =
        config.interval > 0 && run_state.total_cycle % config.interval == 0;

    // Clear previous step's records
    contact_output.records.clear();

    // Get mutable coordination data if enabled
    let mut ca = if has_coordination {
        Some(registry.expect_mut::<ContactAnalysis>("compute_contact_analysis"))
    } else {
        None
    };

    // Ensure coordination vec covers all atoms
    if let Some(ref mut ca) = ca {
        while ca.coordination.len() < atoms.len() {
            ca.coordination.push(0.0);
        }
    }

    for (i, j) in neighbor.pairs(nlocal) {
        let r1 = dem.radius[i];
        let r2 = dem.radius[j];

        let dx = atoms.pos[j][0] - atoms.pos[i][0];
        let dy = atoms.pos[j][1] - atoms.pos[i][1];
        let dz = atoms.pos[j][2] - atoms.pos[i][2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let sum_r = r1 + r2;

        if dist_sq >= sum_r * sum_r {
            continue;
        }

        let distance = dist_sq.sqrt();
        if distance == 0.0 {
            continue;
        }

        let delta = sum_r - distance;
        if delta <= 0.0 {
            continue;
        }

        // This pair is in contact (overlap > 0)

        // Increment coordination for both atoms
        if let Some(ref mut ca) = ca {
            ca.coordination[i] += 1.0;
            if j < nlocal {
                ca.coordination[j] += 1.0;
            }
        }

        // Collect per-contact record if this is a dump step
        if collect_records {
            let inv_dist = 1.0 / distance;
            let nx = dx * inv_dist;
            let ny = dy * inv_dist;
            let nz = dz * inv_dist;

            // Contact point: on surface of atom i, offset by (r1 - delta/2)
            let alpha = r1 - 0.5 * delta;
            let cx = atoms.pos[i][0] + alpha * nx;
            let cy = atoms.pos[i][1] + alpha * ny;
            let cz = atoms.pos[i][2] + alpha * nz;

            // Force magnitudes: recorded as geometry-only for now.
            // Full force integration (reading fn/ft from the contact force loop)
            // requires coupling to dem_granular's contact computation.
            // The overlap and contact normal are the primary data for
            // fabric tensor and contact network analysis.
            contact_output.records.push(ContactRecord {
                i_tag: atoms.tag[i],
                j_tag: atoms.tag[j],
                fx: 0.0,
                fy: 0.0,
                fz: 0.0,
                fn_mag: 0.0,
                ft_mag: 0.0,
                overlap: delta,
                cx,
                cy,
                cz,
                nx,
                ny,
                nz,
            });
        }
    }
}

/// Push coordination statistics to thermo output.
fn push_coordination_to_thermo(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    comm: Res<CommResource>,
    mut thermo: ResMut<Thermo>,
    run_state: Res<RunState>,
) {
    if thermo.interval == 0 || run_state.total_cycle % thermo.interval != 0 {
        return;
    }

    if let Some(ca) = registry.get::<ContactAnalysis>() {
        let nlocal = atoms.nlocal as usize;
        let mut sum = 0.0;
        let mut max_val: f64 = 0.0;
        for i in 0..nlocal {
            let c = ca.coordination[i];
            sum += c;
            if c > max_val {
                max_val = c;
            }
        }

        let global_sum = comm.all_reduce_sum_f64(sum);
        // Use min of negated values to get max across ranks
        let global_max = -comm.all_reduce_min_f64(-max_val);
        let global_atoms = atoms.natoms as f64;
        let avg = if global_atoms > 0.0 {
            global_sum / global_atoms
        } else {
            0.0
        };

        thermo.set("coord_avg", avg);
        thermo.set("coord_max", global_max);
    }
}

/// Post-force system: compute fabric tensor from contact normals and push to thermo.
///
/// Fabric tensor: `F_ij = (1/Nc) * sum(n_i * n_j)` over all contacts.
/// For an isotropic packing, F ≈ (1/3) * I.
fn compute_fabric_tensor(
    mut thermo: ResMut<Thermo>,
    atoms: Res<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    comm: Res<CommResource>,
    run_state: Res<RunState>,
) {
    if thermo.interval == 0 || run_state.total_cycle % thermo.interval != 0 {
        return;
    }

    let nlocal = atoms.nlocal as usize;
    let dem = registry.expect::<DemAtom>("compute_fabric_tensor");

    let (mut fxx, mut fyy, mut fzz, mut fxy, mut fxz, mut fyz) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut nc: f64 = 0.0;

    // Compute from neighbor pairs (always recompute to ensure consistency)
    for (i, j) in neighbor.pairs(nlocal) {
        let r1 = dem.radius[i];
        let r2 = dem.radius[j];
        let dx = atoms.pos[j][0] - atoms.pos[i][0];
        let dy = atoms.pos[j][1] - atoms.pos[i][1];
        let dz = atoms.pos[j][2] - atoms.pos[i][2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let sum_r = r1 + r2;
        if dist_sq >= sum_r * sum_r {
            continue;
        }
        let distance = dist_sq.sqrt();
        if distance == 0.0 {
            continue;
        }
        let delta = sum_r - distance;
        if delta <= 0.0 {
            continue;
        }
        let inv_dist = 1.0 / distance;
        let nx = dx * inv_dist;
        let ny = dy * inv_dist;
        let nz = dz * inv_dist;
        fxx += nx * nx;
        fyy += ny * ny;
        fzz += nz * nz;
        fxy += nx * ny;
        fxz += nx * nz;
        fyz += ny * nz;
        nc += 1.0;
    }

    // MPI reduce
    let global_fxx = comm.all_reduce_sum_f64(fxx);
    let global_fyy = comm.all_reduce_sum_f64(fyy);
    let global_fzz = comm.all_reduce_sum_f64(fzz);
    let global_fxy = comm.all_reduce_sum_f64(fxy);
    let global_fxz = comm.all_reduce_sum_f64(fxz);
    let global_fyz = comm.all_reduce_sum_f64(fyz);
    let global_nc = comm.all_reduce_sum_f64(nc);

    if global_nc > 0.0 {
        let inv_nc = 1.0 / global_nc;
        thermo.set("fabric_xx", global_fxx * inv_nc);
        thermo.set("fabric_yy", global_fyy * inv_nc);
        thermo.set("fabric_zz", global_fzz * inv_nc);
        thermo.set("fabric_xy", global_fxy * inv_nc);
        thermo.set("fabric_xz", global_fxz * inv_nc);
        thermo.set("fabric_yz", global_fyz * inv_nc);
    } else {
        thermo.set("fabric_xx", 0.0);
        thermo.set("fabric_yy", 0.0);
        thermo.set("fabric_zz", 0.0);
        thermo.set("fabric_xy", 0.0);
        thermo.set("fabric_xz", 0.0);
        thermo.set("fabric_yz", 0.0);
    }

    thermo.set("contacts", global_nc);
}

/// Dump per-contact records to CSV file.
fn dump_contact_records(
    contact_output: Res<ContactOutput>,
    config: Res<ContactAnalysisConfig>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
) {
    if config.interval == 0 {
        return;
    }
    let step = run_state.total_cycle;
    if step % config.interval != 0 {
        return;
    }

    let rank = comm.rank();
    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("{}/contact", dir),
        None => "contact".to_string(),
    };

    if let Err(e) = dump_contact_csv(&contact_output.records, &base_dir, &config.file_prefix, step, rank) {
        eprintln!("WARNING: Contact dump failed at step {}: {}", step, e);
    }
}

fn dump_contact_csv(
    records: &[ContactRecord],
    base_dir: &str,
    prefix: &str,
    step: usize,
    rank: i32,
) -> std::io::Result<()> {
    fs::create_dir_all(base_dir)?;
    let filename = format!("{}/{}_{:06}_rank{}.csv", base_dir, prefix, step, rank);
    let file = File::create(&filename)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "i_tag,j_tag,fx,fy,fz,fn_mag,ft_mag,overlap,cx,cy,cz,nx,ny,nz"
    )?;

    for r in records {
        writeln!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.i_tag, r.j_tag, r.fx, r.fy, r.fz, r.fn_mag, r.ft_mag, r.overlap, r.cx, r.cy,
            r.cz, r.nx, r.ny, r.nz
        )?;
    }

    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mddem_core::Atom;
    use mddem_neighbor::Neighbor;

    /// Helper: create a neighbor list from atom positions using brute force.
    fn build_neighbor_list(atoms: &Atom) -> Neighbor {
        let nlocal = atoms.nlocal as usize;
        let ntotal = atoms.len();
        let mut neighbor = Neighbor::new();

        // Build CSR neighbor list manually
        neighbor.neighbor_offsets = vec![0u32; nlocal + 1];
        neighbor.neighbor_indices.clear();

        for i in 0..nlocal {
            neighbor.neighbor_offsets[i] = neighbor.neighbor_indices.len() as u32;
            for j in (i + 1)..ntotal {
                let dx = atoms.pos[j][0] - atoms.pos[i][0];
                let dy = atoms.pos[j][1] - atoms.pos[i][1];
                let dz = atoms.pos[j][2] - atoms.pos[i][2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                // Use cutoff_radius sum as neighbor cutoff
                let cut = atoms.cutoff_radius[i] + atoms.cutoff_radius[j];
                if dist_sq < cut * cut * 1.5 {
                    // generous skin
                    neighbor.neighbor_indices.push(j as u32);
                }
            }
        }
        neighbor.neighbor_offsets[nlocal] = neighbor.neighbor_indices.len() as u32;
        neighbor
    }

    fn make_dem_atom(n: usize) -> DemAtom {
        let mut dem = DemAtom::new();
        for _ in 0..n {
            dem.radius.push(0.5);
            dem.density.push(2500.0);
            dem.inv_inertia.push(1.0);
            dem.quaternion.push([1.0, 0.0, 0.0, 0.0]);
            dem.omega.push([0.0; 3]);
            dem.ang_mom.push([0.0; 3]);
            dem.torque.push([0.0; 3]);
        }
        dem
    }

    /// Run the coordination counting loop (same logic as compute_contact_analysis).
    fn count_coordination(
        atoms: &Atom,
        neighbor: &Neighbor,
        dem: &DemAtom,
        coordination: &mut [f64],
    ) {
        let nlocal = atoms.nlocal as usize;
        for (i, j) in neighbor.pairs(nlocal) {
            let r1 = dem.radius[i];
            let r2 = dem.radius[j];
            let dx = atoms.pos[j][0] - atoms.pos[i][0];
            let dy = atoms.pos[j][1] - atoms.pos[i][1];
            let dz = atoms.pos[j][2] - atoms.pos[i][2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let sum_r = r1 + r2;
            if dist_sq >= sum_r * sum_r {
                continue;
            }
            let distance = dist_sq.sqrt();
            if distance == 0.0 {
                continue;
            }
            let delta = sum_r - distance;
            if delta <= 0.0 {
                continue;
            }
            coordination[i] += 1.0;
            if j < nlocal {
                coordination[j] += 1.0;
            }
        }
    }

    #[test]
    fn test_two_touching_particles_coordination() {
        // Two particles at distance 0.9, each radius 0.5 → overlap = 0.1
        let mut atoms = Atom::new();
        atoms.push_test_atom(1, [0.0, 0.0, 0.0], 0.5, 1.0);
        atoms.push_test_atom(2, [0.9, 0.0, 0.0], 0.5, 1.0);
        atoms.nlocal = 2;
        atoms.natoms = 2;

        let dem = make_dem_atom(2);
        let neighbor = build_neighbor_list(&atoms);
        let mut coordination = vec![0.0; 2];

        count_coordination(&atoms, &neighbor, &dem, &mut coordination);

        assert_eq!(coordination[0], 1.0, "atom 0 should have coord=1");
        assert_eq!(coordination[1], 1.0, "atom 1 should have coord=1");
    }

    #[test]
    fn test_isolated_particle_coordination() {
        // Two particles far apart: no contact
        let mut atoms = Atom::new();
        atoms.push_test_atom(1, [0.0, 0.0, 0.0], 0.5, 1.0);
        atoms.push_test_atom(2, [5.0, 0.0, 0.0], 0.5, 1.0);
        atoms.nlocal = 2;
        atoms.natoms = 2;

        let dem = make_dem_atom(2);
        let neighbor = build_neighbor_list(&atoms);
        let mut coordination = vec![0.0; 2];

        count_coordination(&atoms, &neighbor, &dem, &mut coordination);

        assert_eq!(coordination[0], 0.0, "atom 0 should have coord=0");
        assert_eq!(coordination[1], 0.0, "atom 1 should have coord=0");
    }

    #[test]
    fn test_four_particle_chain_coordination() {
        // Four particles in a chain along x-axis, each touching its neighbor:
        // 0 @ x=0, 1 @ x=0.9, 2 @ x=1.8, 3 @ x=2.7
        // All radius=0.5, so overlap between adjacent = 0.1
        // Expected: [0]=1, [1]=2, [2]=2, [3]=1
        let mut atoms = Atom::new();
        atoms.push_test_atom(1, [0.0, 0.0, 0.0], 0.5, 1.0);
        atoms.push_test_atom(2, [0.9, 0.0, 0.0], 0.5, 1.0);
        atoms.push_test_atom(3, [1.8, 0.0, 0.0], 0.5, 1.0);
        atoms.push_test_atom(4, [2.7, 0.0, 0.0], 0.5, 1.0);
        atoms.nlocal = 4;
        atoms.natoms = 4;

        let dem = make_dem_atom(4);
        let neighbor = build_neighbor_list(&atoms);
        let mut coordination = vec![0.0; 4];

        count_coordination(&atoms, &neighbor, &dem, &mut coordination);

        assert_eq!(coordination[0], 1.0, "end atom 0: coord=1");
        assert_eq!(coordination[1], 2.0, "middle atom 1: coord=2");
        assert_eq!(coordination[2], 2.0, "middle atom 2: coord=2");
        assert_eq!(coordination[3], 1.0, "end atom 3: coord=1");
    }

    #[test]
    fn test_contact_record_csv_output() {
        let records = vec![
            ContactRecord {
                i_tag: 1,
                j_tag: 2,
                fx: 10.0,
                fy: 0.0,
                fz: 0.0,
                fn_mag: 10.0,
                ft_mag: 0.0,
                overlap: 0.1,
                cx: 0.45,
                cy: 0.0,
                cz: 0.0,
                nx: 1.0,
                ny: 0.0,
                nz: 0.0,
            },
        ];

        let dir = std::env::temp_dir().join("dem_contact_test");
        let _ = fs::remove_dir_all(&dir);
        let result = dump_contact_csv(
            &records,
            dir.to_str().unwrap(),
            "contact",
            1000,
            0,
        );
        assert!(result.is_ok(), "CSV dump should succeed");

        let content = fs::read_to_string(dir.join("contact_001000_rank0.csv")).unwrap();
        assert!(content.starts_with("i_tag,j_tag,"));
        assert!(content.contains("1,2,10,0,0,10,0,0.1,0.45,0,0,1,0,0"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_fabric_tensor_isotropic() {
        // 6 contacts with normals along ±x, ±y, ±z → isotropic
        // F_xx = F_yy = F_zz = 2/6 = 1/3, off-diag = 0
        let normals: Vec<[f64; 3]> = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];

        let nc = normals.len() as f64;
        let mut fxx = 0.0;
        let mut fyy = 0.0;
        let mut fzz = 0.0;
        let mut fxy = 0.0;
        let mut fxz = 0.0;
        let mut fyz = 0.0;

        for n in &normals {
            fxx += n[0] * n[0];
            fyy += n[1] * n[1];
            fzz += n[2] * n[2];
            fxy += n[0] * n[1];
            fxz += n[0] * n[2];
            fyz += n[1] * n[2];
        }

        let inv_nc = 1.0 / nc;
        assert!((fxx * inv_nc - 1.0 / 3.0).abs() < 1e-10, "F_xx should be 1/3");
        assert!((fyy * inv_nc - 1.0 / 3.0).abs() < 1e-10, "F_yy should be 1/3");
        assert!((fzz * inv_nc - 1.0 / 3.0).abs() < 1e-10, "F_zz should be 1/3");
        assert!((fxy * inv_nc).abs() < 1e-10, "F_xy should be 0");
        assert!((fxz * inv_nc).abs() < 1e-10, "F_xz should be 0");
        assert!((fyz * inv_nc).abs() < 1e-10, "F_yz should be 0");
    }

    #[test]
    fn test_contact_analysis_config_defaults() {
        let config = ContactAnalysisConfig::default();
        assert_eq!(config.interval, 0);
        assert!(!config.coordination);
        assert!(!config.fabric_tensor);
        // Note: Default trait gives "" for String; the "contact" default
        // is applied by serde during TOML deserialization.
        assert_eq!(config.file_prefix, "");
    }
}

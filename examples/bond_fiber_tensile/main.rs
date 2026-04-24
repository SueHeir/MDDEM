//! BPM fiber tensile test — recover Young's modulus from σ(ε).
//!
//! Pulls an 11-sphere fiber (10 bonds) along +x, fixes the left end, and
//! records the tension carried by each bond every N steps. A Python script
//! fits σ vs ε and compares the slope to the input Young's modulus.
//!
//! Run:
//! ```bash
//! cargo run --release --example bond_fiber_tensile --no-default-features -- \
//!     examples/bond_fiber_tensile/config.toml
//! ```

use mddem::prelude::*;
use mddem::dem_atom::DemAtom;
use mddem::dem_bond::BondConfig;
use mddem::mddem_core::BondStore;
use mddem::mddem_fixes::FixesPlugin;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{BufWriter, Write as IoWrite};

/// Holds the CSV writer + cached geometry so the recorder runs cheaply.
struct Recorder {
    writer: Option<BufWriter<File>>,
    /// Initial centre-to-centre fiber length (x_10 − x_0) at setup time.
    length0: f64,
    /// Cross-sectional area of each bond, for stress = F/A.
    area: f64,
    /// Effective axial stiffness used by the bond force (E·A/L per bond).
    k_n: f64,
    /// Only record every `record_every` steps.
    record_every: usize,
    initialized: bool,
}

impl Recorder {
    fn new() -> Self {
        Recorder {
            writer: None,
            length0: 0.0,
            area: 0.0,
            k_n: 0.0,
            record_every: 100,
            initialized: false,
        }
    }
}

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins)
        .add_plugins(GranularDefaultPlugins)
        .add_plugins(FixesPlugin)
        .add_plugins(DemBondPlugin);

    app.add_resource(Recorder::new());
    app.add_update_system(record_stress_strain, ParticleSimScheduleSet::PostFinalIntegration);

    app.start();
}

/// Finds the local atom index for a given global tag. Returns `None` if not local.
fn index_of_tag(atoms: &Atom, tag: u32) -> Option<usize> {
    let nlocal = atoms.nlocal as usize;
    (0..nlocal).find(|&i| atoms.tag[i] == tag)
}

/// System: measures fiber tension and global strain every `record_every` steps.
///
/// Uses the **middle bond** (between local tags 6↔7) as the tension probe:
/// F_mid = K_n · δ_mid, σ_mid = F_mid / A. Global strain ε uses the end-to-end
/// displacement between tags 1 and 11 (1-based CSV tags).
fn record_stress_strain(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    bond_config: Res<BondConfig>,
    run_state: Res<RunState>,
    input: Res<Input>,
    mut recorder: ResMut<Recorder>,
) {
    // CSV insertion assigns tags 0..=10 (first particle gets max_tag = 0).
    let tag_left = 0u32;
    let tag_right = 10u32;
    let tag_mid_a = 5u32; // middle bond is 5↔6
    let tag_mid_b = 6u32;

    let i_left = match index_of_tag(&atoms, tag_left) { Some(i) => i, None => return };
    let i_right = match index_of_tag(&atoms, tag_right) { Some(i) => i, None => return };
    let i_mid_a = match index_of_tag(&atoms, tag_mid_a) { Some(i) => i, None => return };
    let i_mid_b = match index_of_tag(&atoms, tag_mid_b) { Some(i) => i, None => return };

    if !recorder.initialized {
        // Cache geometry and open output file once (first time atom 0 is seen).
        let dem = registry.expect::<DemAtom>("record_stress_strain");
        let r_i = dem.radius[i_mid_a];
        let r_j = dem.radius[i_mid_b];
        let r_b = bond_config.bond_radius_ratio * r_i.min(r_j);
        let area = PI * r_b * r_b;

        let dx = atoms.pos[i_right][0] - atoms.pos[i_left][0];
        let length0 = dx;

        // K_n per bond: material mode prefers E·A/L; else direct stiffness.
        let bond_len_mid = {
            let bonds = registry.expect::<BondStore>("record_stress_strain");
            bonds.bonds[i_mid_a]
                .iter()
                .find(|b| b.partner_tag == tag_mid_b)
                .map(|b| b.r0)
                .expect("middle bond must exist")
        };
        let k_n = match bond_config.youngs_modulus {
            Some(e) => e * area / bond_len_mid,
            None => bond_config.normal_stiffness,
        };

        let out_dir = input
            .output_dir
            .clone()
            .unwrap_or_else(|| "examples/bond_fiber_tensile".to_string());
        fs::create_dir_all(format!("{}/data", out_dir)).ok();
        let path = format!("{}/data/fiber_tensile.csv", out_dir);
        let mut w = BufWriter::new(
            File::create(&path).unwrap_or_else(|e| panic!("Cannot create {}: {}", path, e)),
        );
        writeln!(
            w,
            "step,t,length,delta_L,strain_global,strain_mid,force_mid,stress_mid,bond_radius,area,k_n,length0"
        )
        .unwrap();

        recorder.writer = Some(w);
        recorder.length0 = length0;
        recorder.area = area;
        recorder.k_n = k_n;
        recorder.initialized = true;

        println!("=== BPM Fiber Tensile Test ===");
        println!("  L0 = {:.6e} m    (11-sphere fiber, 10 bonds)", length0);
        println!("  r_b = {:.6e} m,  A = {:.6e} m²", r_b, area);
        println!("  K_n per bond = {:.6e} N/m", k_n);
        println!("  σ(ε) written to {}", path);
    }

    let step = run_state.total_cycle;
    if step % recorder.record_every != 0 {
        return;
    }

    let dt = atoms.dt;
    let t = step as f64 * dt;
    let length = atoms.pos[i_right][0] - atoms.pos[i_left][0];
    let delta_l = length - recorder.length0;
    let strain_global = delta_l / recorder.length0;

    // Middle-bond stretch and derived force/stress.
    let bonds = registry.expect::<BondStore>("record_stress_strain");
    let mid_bond = bonds.bonds[i_mid_a]
        .iter()
        .find(|b| b.partner_tag == tag_mid_b)
        .expect("middle bond must exist");
    let dx = atoms.pos[i_mid_b][0] - atoms.pos[i_mid_a][0];
    let dy = atoms.pos[i_mid_b][1] - atoms.pos[i_mid_a][1];
    let dz = atoms.pos[i_mid_b][2] - atoms.pos[i_mid_a][2];
    let dist = (dx*dx + dy*dy + dz*dz).sqrt();
    let delta_mid = dist - mid_bond.r0;
    let strain_mid = delta_mid / mid_bond.r0;
    let force_mid = recorder.k_n * delta_mid;
    let stress_mid = force_mid / recorder.area;

    let area = recorder.area;
    let k_n = recorder.k_n;
    let length0 = recorder.length0;
    let r_b = (area / PI).sqrt();
    if let Some(ref mut w) = recorder.writer {
        writeln!(
            w,
            "{},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e}",
            step, t, length, delta_l, strain_global, strain_mid,
            force_mid, stress_mid, r_b, area, k_n, length0,
        )
        .ok();
    }
}

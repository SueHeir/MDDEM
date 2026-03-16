use mddem::md_lj::{self, LJPairTable};
use mddem::prelude::*;
use mddem_derive::AtomData;

// ── Per-atom energy extension ────────────────────────────────────────────────

/// Per-atom potential energy tracked via the `AtomData` derive macro.
///
/// - `#[reverse]` — ghost contributions are summed back to the owning rank.
/// - `#[zero]`    — values are zeroed each step before the force/energy loop.
#[derive(AtomData)]
pub struct LJAtomEnergy {
    #[reverse]
    #[zero]
    pub pe: Vec<f64>,
}

impl Default for LJAtomEnergy {
    fn default() -> Self {
        Self::new()
    }
}

impl LJAtomEnergy {
    pub fn new() -> Self {
        LJAtomEnergy { pe: Vec::new() }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

pub struct LJAtomEnergyPlugin;

impl Plugin for LJAtomEnergyPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(AtomPlugin);
        register_atom_data!(app, LJAtomEnergy::new());

        Config::load::<md_lj::LJConfig>(app, "lj");

        app.add_plugins(VirialStressPlugin);
        let mut default_table = PairCoeffTable::new(1, md_lj::LJPairCoeffs::default());
        default_table.set(0, 0, md_lj::LJPairCoeffs::from_params(1.0, 1.0, 2.5));
        app.add_resource(LJPairTable(default_table));
        app.add_resource(md_lj::LJTailCorrections::default())
            .add_setup_system(md_lj::build_lj_pair_table, ScheduleSetupSet::Setup)
            .add_setup_system(md_lj::setup_lj_tails, ScheduleSetupSet::PostSetup)
            .add_update_system(
                lj_force_with_energy.label("lj"),
                ScheduleSet::Force,
            )
            .add_update_system(output_per_atom_energy, ScheduleSet::PostForce);
    }
}

// ── Force + per-atom energy system ──────────────────────────────────────────

fn lj_force_with_energy(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    lj: Res<md_lj::LJConfig>,
    registry: Res<AtomDataRegistry>,
) {
    let cutoff2 = (lj.cutoff * lj.sigma).powi(2);
    let sigma6 = (lj.sigma * lj.sigma).powi(3);
    let lj1: f64 = 48.0 * lj.epsilon * sigma6 * sigma6; // force: 48*eps*sigma^12
    let lj2: f64 = 24.0 * lj.epsilon * sigma6;           // force: 24*eps*sigma^6
    let lj3: f64 = 4.0 * lj.epsilon * sigma6 * sigma6;   // energy: 4*eps*sigma^12
    let lj4: f64 = 4.0 * lj.epsilon * sigma6;            // energy: 4*eps*sigma^6

    let nlocal = atoms.nlocal as usize;

    let pos_ptr = atoms.pos.as_ptr();
    let force_ptr = atoms.force.as_mut_ptr();
    let offsets_ptr = neighbor.neighbor_offsets.as_ptr();
    let indices_ptr = neighbor.neighbor_indices.as_ptr();

    let mut pe_data = registry.get_mut::<LJAtomEnergy>().unwrap();
    pe_data.pe.resize(atoms.len(), 0.0);
    let pe_ptr = pe_data.pe.as_mut_ptr();

    for i in 0..nlocal {
        let pi = unsafe { *pos_ptr.add(i) };
        let mut fi = unsafe { *force_ptr.add(i) };
        let start = unsafe { *offsets_ptr.add(i) } as usize;
        let end = unsafe { *offsets_ptr.add(i + 1) } as usize;

        for k in start..end {
            let j = unsafe { *indices_ptr.add(k) } as usize;
            let pj = unsafe { *pos_ptr.add(j) };
            let dx = pj[0] - pi[0];
            let dy = pj[1] - pi[1];
            let dz = pj[2] - pi[2];
            let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));

            if r2 >= cutoff2 {
                continue;
            }

            let r2inv = 1.0 / r2;
            let r6inv = r2inv * r2inv * r2inv;

            // Force
            let fpair = r2inv * r6inv * lj1.mul_add(r6inv, -lj2);
            fi[0] = (-fpair).mul_add(dx, fi[0]);
            fi[1] = (-fpair).mul_add(dy, fi[1]);
            fi[2] = (-fpair).mul_add(dz, fi[2]);
            let fj = unsafe { &mut *force_ptr.add(j) };
            fj[0] = fpair.mul_add(dx, fj[0]);
            fj[1] = fpair.mul_add(dy, fj[1]);
            fj[2] = fpair.mul_add(dz, fj[2]);

            // Energy: 4*eps*[(sigma/r)^12 - (sigma/r)^6], split half to each atom
            let pair_pe = r6inv * lj3.mul_add(r6inv, -lj4);
            let half_pe = 0.5 * pair_pe;
            unsafe {
                *pe_ptr.add(i) += half_pe;
                *pe_ptr.add(j) += half_pe;
            }
        }
        unsafe { *force_ptr.add(i) = fi };
    }
}

// ── Sum per-atom PE and push to thermo ──────────────────────────────────────

fn output_per_atom_energy(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    mut thermo: ResMut<Thermo>,
) {
    if !run_state.total_cycle.is_multiple_of(thermo.interval) {
        return;
    }

    let pe_data = registry.get::<LJAtomEnergy>().unwrap();
    let nlocal = atoms.nlocal as usize;
    let local_pe: f64 = pe_data.pe[..nlocal].iter().sum();
    let global_pe = comm.all_reduce_sum_f64(local_pe);
    thermo.set("pe", global_pe);
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let mut app = App::new();
    app.add_plugins(CorePlugins);
    app.add_plugins(LatticePlugin);
    app.add_plugins(LJAtomEnergyPlugin);
    app.add_plugins(NoseHooverPlugin);
    app.add_plugins(MeasurePlugin);
    app.start();
}

# dem_contact_analysis

Contact network analysis for DEM simulations: per-atom coordination number, rattler detection, per-contact CSV dumps, and fabric tensor computation.

## What It Does

`dem_contact_analysis` provides a `ContactAnalysisPlugin` that analyzes contact networks in DEM systems. It computes:

- **Coordination number**: count of active contacts per particle (via `ContactAnalysis` AtomData)
- **Rattlers**: particles with < 4 contacts (mechanically unstable in 3D)
- **Per-contact CSV output**: geometric data (position, overlap, normal) for every contact
- **Fabric tensor**: second-order tensor F_ij = (1/N_c) Σ n_i·n_j measuring directional isotropy

## Key Types

- `ContactAnalysisConfig`: TOML configuration with `interval`, `coordination`, `rattlers`, `fabric_tensor`, `file_prefix` fields
- `ContactAnalysis`: per-atom data (stores coordination number); implements `AtomData`
- `ContactRecord`: single contact's geometry (tags, overlap, contact point, normal)
- `ContactOutput`: resource holding contact records for the current step
- `ContactAnalysisPlugin`: registers all analysis systems

## Configuration Example

```toml
[contact_analysis]
interval = 1000          # dump per-contact CSV every N steps (0 = disabled)
coordination = true      # compute coordination number
rattlers = true          # detect rattler particles
fabric_tensor = true     # output fabric tensor components to thermo
file_prefix = "contact"  # CSV filename prefix
```

## Usage

Add `ContactAnalysisPlugin` to your app:

```rust
use dem_contact_analysis::ContactAnalysisPlugin;

app.add_plugin(ContactAnalysisPlugin);
```

Data flows to thermo output:
- `coord_avg`, `coord_max`, `coord_min` — coordination statistics
- `n_rattlers`, `rattler_fraction` — (when rattlers enabled)
- `fabric_xx`, `fabric_yy`, `fabric_zz`, `fabric_xy`, `fabric_xz`, `fabric_yz`, `contacts` — fabric tensor components

CSV records (when `interval > 0`) are written to `<output_dir>/contact/<prefix>_<step>_rank<rank>.csv` with columns: `i_tag`, `j_tag`, `overlap`, `cx`, `cy`, `cz`, `nx`, `ny`, `nz`.

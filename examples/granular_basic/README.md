# Granular Basic

Basic 3D granular gas simulation with 500 particles in a periodic box.

## Run

```bash
# Single-process
cargo run --example granular_basic -- examples/granular_basic/config.toml

# With MPI
cargo build-examples
mpiexec -n 4 ./target/release/examples/granular_basic examples/granular_basic/config.toml
```

## Parameters

| Parameter | Value |
|-----------|-------|
| Particles | 500 |
| Radius | 0.001 m |
| Density | 2500 kg/m^3 |
| Young's modulus | 8.7 GPa |
| Poisson ratio | 0.3 |
| Initial velocity | 0.5 m/s (Gaussian per component) |
| Restitution | 0.95 |
| Friction | 0.4 |
| Domain | 0.025 x 0.025 x 0.025 m (periodic) |
| Steps | 5,000,000 |
| Thermo interval | 500 |

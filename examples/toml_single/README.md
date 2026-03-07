# Programmatic Config

Quick test demonstrating the Rust API tier: 500 particles, 10,000 steps, with the entire configuration built programmatically in `main.rs` — no TOML file needed.

This example shows how to construct a `Config` table directly in Rust code, useful for:
- Embedding MDDEM as a library in a larger application
- Generating configurations from code (parameter sweeps, automated testing)
- Simulations that need logic beyond what TOML can express

## Run

```bash
# Single-process (no config file needed — config is built in code)
cargo run --example toml_single

# With MPI
cargo build-examples
mpiexec -n 4 ./target/release/examples/toml_single
```

A `config.toml` with equivalent parameters is also included for reference.

## Parameters

| Parameter | Value |
|-----------|-------|
| Particles | 500 |
| Radius | 0.001 m |
| Density | 2500 kg/m^3 |
| Young's modulus | 8.7 GPa |
| Restitution | 0.95 |
| Friction | 0.4 |
| Domain | 0.025^3 m (periodic) |
| Steps | 10,000 |
| Thermo interval | 100 |

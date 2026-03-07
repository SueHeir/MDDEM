# TOML Single Run

Single granular simulation configured via TOML. A quick test case with 500 particles and 10,000 steps.

## Run

```bash
mpiexec -n 4 ./target/release/MDDEM ./examples/toml_single/config.toml
```

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

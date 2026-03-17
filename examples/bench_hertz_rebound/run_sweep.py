#!/usr/bin/env python3
"""
Run parameter sweep for Hertz rebound benchmark.

Generates TOML configs for each (velocity, COR) combination, runs the
simulation, and collects results into a single CSV file.

Usage (from repo root):
    python3 examples/bench_hertz_rebound/run_sweep.py
"""

import os
import subprocess
import sys
import csv

# Parameter sweep
VELOCITIES = [0.1, 0.5, 1.0, 2.0]  # m/s
CORS = [0.5, 0.7, 0.9, 0.95]

# Material properties (must match config.toml)
RADIUS = 0.005        # m
DENSITY = 2500.0      # kg/m^3
YOUNGS_MOD = 70.0e9   # Pa
POISSON_RATIO = 0.22

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

TOML_TEMPLATE = """\
# Auto-generated config for Hertz rebound sweep
# v0 = {v0} m/s, COR = {cor}

[comm]
processors_x = 1
processors_y = 1
processors_z = 1

[domain]
x_low = -0.01
x_high = 0.01
y_low = -0.01
y_high = 0.01
z_low = 0.0
z_high = 0.1
periodic_x = false
periodic_y = false
periodic_z = false

[neighbor]
skin_fraction = 1.1
bin_size = 0.015
every = 1

[[dem.materials]]
name = "glass"
youngs_mod = {youngs_mod}
poisson_ratio = {poisson_ratio}
restitution = {cor}
friction = 0.0

[[particles.insert]]
material = "glass"
count = 1
radius = {radius}
density = {density}
velocity_z = -{v0}
region = {{ type = "block", min = [-0.001, -0.001, 0.007], max = [0.001, 0.001, 0.008] }}

[[wall]]
point_x = 0.0
point_y = 0.0
point_z = 0.0
normal_x = 0.0
normal_y = 0.0
normal_z = 1.0
material = "glass"

[output]
dir = "{output_dir}"

[run]
steps = {steps}
thermo = 5000
"""


def main():
    os.chdir(REPO_ROOT)

    # Build first
    print("Building in release mode...")
    result = subprocess.run(
        ["cargo", "build", "--release", "--example", "bench_hertz_rebound",
         "--no-default-features"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Build failed:")
        print(result.stderr)
        sys.exit(1)
    print("Build succeeded.\n")

    # Prepare output
    data_dir = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    sweep_file = os.path.join(data_dir, "sweep_results.csv")

    results = []

    for cor in CORS:
        for v0 in VELOCITIES:
            tag = f"v{v0}_cor{cor}"
            case_dir = os.path.join(SCRIPT_DIR, "sweep", tag)
            os.makedirs(case_dir, exist_ok=True)

            # Estimate steps needed: particle starts at z~0.007, radius=0.005,
            # so gap ~2mm. No gravity. Need travel_time/dt + contact_time/dt + rebound margin
            fall_dist = 0.003  # m (z=0.007, surface at z=0.005, so ~2mm gap)
            fall_time = fall_dist / v0
            # Rayleigh dt estimate
            g = YOUNGS_MOD / (2.0 * (1.0 + POISSON_RATIO))
            alpha = 0.1631 * POISSON_RATIO + 0.876605
            dt_rayleigh = 3.14159 * RADIUS / alpha * (DENSITY / g) ** 0.5
            dt = dt_rayleigh * 0.15
            total_time = fall_time * 3.0  # generous margin
            steps = int(total_time / dt) + 10000

            config_path = os.path.join(case_dir, "config.toml")
            output_dir = case_dir

            with open(config_path, "w") as f:
                f.write(TOML_TEMPLATE.format(
                    v0=v0, cor=cor,
                    youngs_mod=f"{YOUNGS_MOD:.1e}",
                    poisson_ratio=POISSON_RATIO,
                    radius=RADIUS,
                    density=DENSITY,
                    output_dir=output_dir,
                    steps=steps,
                ))

            print(f"Running: v0={v0} m/s, COR={cor} ({steps} steps)...")
            result = subprocess.run(
                ["cargo", "run", "--release", "--example", "bench_hertz_rebound",
                 "--no-default-features", "--", config_path],
                capture_output=True, text=True, timeout=300,
            )

            if result.returncode != 0:
                print(f"  FAILED: {result.stderr[-500:]}")
                continue

            # Read results
            results_file = os.path.join(output_dir, "data", "rebound_results.csv")
            if not os.path.exists(results_file):
                print(f"  WARNING: No results file found at {results_file}")
                # Check stdout for clues
                for line in result.stdout.split('\n'):
                    if 'Results' in line or 'COR' in line or 'ERROR' in line:
                        print(f"    {line}")
                continue

            with open(results_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["input_v0"] = str(v0)
                    row["input_cor"] = str(cor)
                    results.append(row)

            print(f"  Done. COR_measured={results[-1]['cor_measured']}")

    # Write combined results
    if results:
        with open(sweep_file, "w", newline="") as f:
            fieldnames = ["input_v0", "input_cor", "v_impact", "v_rebound",
                          "cor_measured", "contact_time", "max_overlap", "dt",
                          "radius", "density"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSweep results written to: {sweep_file}")
        print(f"Total cases: {len(results)}/{len(CORS)*len(VELOCITIES)}")
    else:
        print("\nERROR: No results collected!")
        sys.exit(1)


if __name__ == "__main__":
    main()

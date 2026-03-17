#!/usr/bin/env python3
"""
Run the full lunar regolith cohesive angle of repose benchmark.

Generates TOML configs for multiple (gravity, surface_energy) combinations,
runs each simulation, measures the pile profile from the final particle
dump, and saves results to results.csv.

Usage:
    cd <repo_root>
    python examples/bench_lunar_regolith/run_benchmark.py

Requires: numpy
"""

import os
import sys
import subprocess
import glob
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Parametric study: (label, gz [m/s^2], surface_energy [J/m^2])
CASES = [
    ("earth_gamma0.000", -9.81, 0.0),
    ("earth_gamma0.005", -9.81, 0.005),
    ("earth_gamma0.020", -9.81, 0.02),
    ("earth_gamma0.050", -9.81, 0.05),
    ("lunar_gamma0.000", -1.62, 0.0),
    ("lunar_gamma0.005", -1.62, 0.005),
    ("lunar_gamma0.020", -1.62, 0.02),
    ("lunar_gamma0.050", -1.62, 0.05),
]

# Physical constants for Bond number calculation
RADIUS = 0.001       # [m]
DENSITY = 1500.0     # [kg/m^3]
MASS = DENSITY * (4.0 / 3.0) * np.pi * RADIUS**3
R_EFF = RADIUS / 2.0  # Effective radius for equal-size particles
COLUMN_HALF_WIDTH = 0.012  # [m] Initial column half-width

CONFIG_TEMPLATE = """\
# Auto-generated config for: {label}
# Gravity: {gz} m/s^2, Surface energy: {surface_energy} J/m^2
# Bond number: {bond_number:.3f}

[comm]
processors_x = 1
processors_y = 1
processors_z = 1

[domain]
x_low = -0.06
x_high = 0.06
y_low = 0.0
y_high = 0.006
z_low = 0.0
z_high = 0.10
periodic_x = false
periodic_y = true
periodic_z = false

[neighbor]
skin_fraction = 1.2
bin_size = 0.004
every = 5
rebuild_on_pbc_wrap = true

[gravity]
gx = 0.0
gy = 0.0
gz = {gz}

[dem]
contact_model = "hertz"

[[dem.materials]]
name = "regolith"
youngs_mod = 5e6
poisson_ratio = 0.3
restitution = 0.3
friction = 0.5
rolling_friction = 0.1
surface_energy = {surface_energy}

[[particles.insert]]
material = "regolith"
count = 350
radius = 0.001
density = 1500.0
region = {{ type = "block", min = [-0.012, 0.0, 0.002], max = [0.012, 0.006, 0.09] }}

[[wall]]
point_z = 0.0
normal_z = 1.0
material = "regolith"

[[wall]]
point_x = -0.06
normal_x = 1.0
material = "regolith"

[[wall]]
point_x = 0.06
normal_x = -1.0
material = "regolith"

[[wall]]
point_z = 0.095
normal_z = -1.0
material = "regolith"

[[wall]]
point_x = -0.012
normal_x = 1.0
material = "regolith"
name = "column_left"

[[wall]]
point_x = 0.012
normal_x = -1.0
material = "regolith"
name = "column_right"

[output]
dir = "{output_dir}"

[[run]]
name = "settling"
steps = 50000
thermo = 10000
dump_interval = 50000

[[run]]
name = "collapse"
steps = 100000
thermo = 10000
dump_interval = 100000
"""


def compute_bond_number(surface_energy, gz):
    """Compute Bo = F_adhesion / (m * |g|) for equal spheres with JKR."""
    if abs(gz) < 1e-10 or surface_energy <= 0:
        return 0.0
    f_adhesion = 1.5 * np.pi * surface_energy * R_EFF
    return f_adhesion / (MASS * abs(gz))


def generate_config(label, gz, surface_energy):
    """Generate a TOML config file for a single case."""
    output_dir = os.path.join(SCRIPT_DIR, f"output_{label}")
    bond_number = compute_bond_number(surface_energy, gz)
    config_path = os.path.join(SCRIPT_DIR, f"config_{label}.toml")

    config_text = CONFIG_TEMPLATE.format(
        label=label,
        gz=gz,
        surface_energy=surface_energy,
        bond_number=bond_number,
        output_dir=output_dir,
    )

    with open(config_path, "w") as f:
        f.write(config_text)

    return config_path, output_dir, bond_number


def run_simulation(config_path):
    """Run a single MDDEM simulation."""
    cmd = [
        "cargo", "run", "--release", "--no-default-features",
        "--example", "bench_lunar_regolith",
        "--", config_path,
    ]
    print(f"  Running: {os.path.basename(config_path)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  WARNING: Simulation failed (rc={result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr[-300:]}")
        return False
    return True


def find_last_dump(output_dir):
    """Find the last dump CSV file in the output directory."""
    pattern = os.path.join(output_dir, "dump", "dump_*_rank0.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[-1]


def measure_pile_profile(dump_file):
    """
    Measure the pile profile from a particle dump CSV.

    For a column collapse, measures:
    - Peak height of the deposit
    - Runout distance from initial column edge
    - Effective angle = arctan(peak_height / runout_from_column_edge)

    Returns: (angle_deg, peak_height, runout)
    """
    try:
        data = np.genfromtxt(dump_file, delimiter=",", names=True)
    except Exception:
        return None, None, None

    try:
        x = np.array(data["x"])
        z = np.array(data["z"])
        r = np.array(data["radius"]) if "radius" in data.dtype.names else np.full(len(x), 0.001)
    except (ValueError, KeyError):
        return None, None, None

    if len(x) < 10:
        return None, None, None

    # Surface profile: bin by x, find max z in each bin
    n_bins = 30
    x_min, x_max = x.min() - 0.001, x.max() + 0.001
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    surface_z = np.full(n_bins, np.nan)

    for b in range(n_bins):
        mask = (x >= bin_edges[b]) & (x < bin_edges[b + 1])
        if np.any(mask):
            idx_max = np.argmax(z[mask])
            surface_z[b] = z[mask][idx_max] + r[mask][idx_max]

    valid = ~np.isnan(surface_z)
    if np.sum(valid) < 5:
        return None, None, None

    xc = bin_centers[valid]
    zc = surface_z[valid]

    # Peak height
    peak_z = np.max(zc)
    floor_z = np.min(zc)
    pile_height = peak_z - floor_z

    if pile_height < 0.002:  # Less than 1 particle diameter
        return None, None, None

    # Runout: distance from column edge to the furthest particle
    runout = max(abs(x.max()) - COLUMN_HALF_WIDTH,
                 abs(x.min()) - COLUMN_HALF_WIDTH)
    runout = max(runout, 0.001)

    # Effective angle: tan(alpha) = pile_height / runout
    angle = np.degrees(np.arctan(pile_height / runout))

    return angle, pile_height, runout


def main():
    print("=" * 60)
    print("Lunar Regolith Cohesive Angle of Repose Benchmark")
    print("=" * 60)

    results = []

    for label, gz, gamma in CASES:
        print(f"\n--- Case: {label} ---")
        config_path, output_dir, bond_number = generate_config(label, gz, gamma)
        print(f"  Bond number: {bond_number:.3f}")

        success = run_simulation(config_path)
        if not success:
            print(f"  SKIPPED (simulation failed)")
            results.append((label, gz, gamma, bond_number, np.nan, np.nan, np.nan))
            continue

        dump_file = find_last_dump(output_dir)
        if dump_file is None:
            print(f"  WARNING: No dump file found in {output_dir}")
            results.append((label, gz, gamma, bond_number, np.nan, np.nan, np.nan))
            continue

        angle, height, runout = measure_pile_profile(dump_file)
        if angle is None:
            print(f"  WARNING: Could not measure pile profile")
            results.append((label, gz, gamma, bond_number, np.nan, np.nan, np.nan))
        else:
            print(f"  Angle: {angle:.1f} deg, Height: {height*1000:.1f} mm, Runout: {runout*1000:.1f} mm")
            results.append((label, gz, gamma, bond_number, angle, height, runout))

    # Save results
    results_file = os.path.join(SCRIPT_DIR, "results.csv")
    with open(results_file, "w") as f:
        f.write("label,gravity,surface_energy,bond_number,angle_deg,pile_height,runout\n")
        for label, gz, gamma, bo, angle, height, runout in results:
            if np.isnan(angle) if isinstance(angle, float) else angle is None:
                f.write(f"{label},{gz},{gamma},{bo:.6f},NaN,NaN,NaN\n")
            else:
                f.write(f"{label},{gz},{gamma},{bo:.6f},{angle:.2f},{height:.6f},{runout:.6f}\n")

    print(f"\nResults saved to: {results_file}")
    print("\nSummary:")
    print(f"{'Label':<25} {'g':>6} {'gamma':>8} {'Bo':>8} {'Angle':>8} {'Height':>8} {'Runout':>8}")
    print("-" * 80)
    for label, gz, gamma, bo, angle, height, runout in results:
        a_str = f"{angle:.1f}" if angle is not None and not np.isnan(angle) else "N/A"
        h_str = f"{height*1000:.1f}" if height is not None and not np.isnan(height) else "N/A"
        r_str = f"{runout*1000:.1f}" if runout is not None and not np.isnan(runout) else "N/A"
        print(f"{label:<25} {gz:>6.2f} {gamma:>8.4f} {bo:>8.3f} {a_str:>8} {h_str:>8} {r_str:>8}")

    return results


if __name__ == "__main__":
    main()

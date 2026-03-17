#!/usr/bin/env python3
"""
Run the full lunar regolith cohesive angle of repose benchmark.

Generates TOML configs for multiple (gravity, surface_energy) combinations,
runs each simulation with funnel-pour particle insertion, measures the pile
profile from the final particle dump, and saves results to results.csv.

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
boundary_z = "shrink-wrap"

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

# Rate-based insertion: drop particles from narrow slot above center
[[particles.insert]]
material = "regolith"
radius = 0.001
density = 1500.0
rate = 10
rate_interval = 500
rate_limit = 350
velocity_z = -0.5
region = {{ type = "block", min = [-0.005, 0.0, 0.04], max = [0.005, 0.006, 0.06] }}

# Floor wall
[[wall]]
point_z = 0.0
normal_z = 1.0
material = "regolith"

[output]
dir = "{output_dir}"

[[run]]
name = "pour"
steps = 200000
thermo = 50000
dt = 5e-6
dump_interval = 200000
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
            print(f"  stderr: {result.stderr[-500:]}")
        return False
    return True


def find_last_dump(output_dir):
    """Find the last dump CSV file in the output directory."""
    pattern = os.path.join(output_dir, "dump", "dump_*_rank0.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[-1]


def measure_pile_angle(dump_file):
    """
    Measure the pile angle from a particle dump CSV using surface profile fitting.

    Method:
    1. Bin particles by |x| (exploit symmetry about x=0)
    2. Find surface height in each bin (max z + radius)
    3. Fit a line to the surface profile
    4. Angle = atan(|slope|)

    Returns: (angle_deg, pile_height, base_radius)
    """
    try:
        data = np.genfromtxt(dump_file, delimiter=",", names=True)
    except Exception:
        return None, None, None

    try:
        x = np.array(data["x"])
        z = np.array(data["z"])
        r = np.array(data["radius"]) if "radius" in data.dtype.names else np.full(len(x), RADIUS)
    except (ValueError, KeyError):
        return None, None, None

    if len(x) < 10:
        return None, None, None

    # Use |x| to exploit symmetry
    abs_x = np.abs(x)
    surface_top = z + r  # Top of each particle

    # Bin by |x|
    n_bins = 20
    x_max = np.percentile(abs_x, 98) + RADIUS  # Avoid outliers
    bin_edges = np.linspace(0, x_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    surface_z = np.full(n_bins, np.nan)

    for b in range(n_bins):
        mask = (abs_x >= bin_edges[b]) & (abs_x < bin_edges[b + 1])
        if np.sum(mask) >= 2:  # Need at least 2 particles in a bin
            surface_z[b] = np.max(surface_top[mask])

    valid = ~np.isnan(surface_z)
    if np.sum(valid) < 3:
        return None, None, None

    xc = bin_centers[valid]
    zc = surface_z[valid]

    # Pile height (peak near center minus floor)
    floor_z = RADIUS  # Particles rest on floor at z=radius
    peak_z = np.max(zc)
    pile_height = peak_z - floor_z

    if pile_height < 3 * RADIUS:  # Less than 1.5 particle diameters
        return None, None, None

    # Fit line to surface profile: z = a * |x| + b
    # Expect negative slope (height decreases with distance from center)
    coeffs = np.polyfit(xc, zc, 1)
    slope = coeffs[0]

    # Angle = atan(|slope|)
    angle_deg = np.degrees(np.arctan(abs(slope)))

    # Base radius: where the fitted line crosses the floor level
    if abs(slope) > 1e-6:
        base_radius = (coeffs[1] - floor_z) / abs(slope)
    else:
        base_radius = x_max

    return angle_deg, pile_height, base_radius


def main():
    print("=" * 60)
    print("Lunar Regolith Cohesive Angle of Repose Benchmark")
    print("Method: Funnel pour with rate-based insertion")
    print("=" * 60)

    results = []

    for label, gz, gamma in CASES:
        print(f"\n--- Case: {label} ---")
        config_path, output_dir, bond_number = generate_config(label, gz, gamma)
        print(f"  Bond number: {bond_number:.3f}")

        success = run_simulation(config_path)
        if not success:
            print("  SKIPPED (simulation failed)")
            results.append((label, gz, gamma, bond_number, np.nan, np.nan, np.nan))
            continue

        dump_file = find_last_dump(output_dir)
        if dump_file is None:
            print(f"  WARNING: No dump file found in {output_dir}")
            results.append((label, gz, gamma, bond_number, np.nan, np.nan, np.nan))
            continue

        angle, height, base_r = measure_pile_angle(dump_file)
        if angle is None:
            print("  WARNING: Could not measure pile angle")
            results.append((label, gz, gamma, bond_number, np.nan, np.nan, np.nan))
        else:
            print(f"  Angle: {angle:.1f} deg, Height: {height*1000:.1f} mm, Base: {base_r*1000:.1f} mm")
            results.append((label, gz, gamma, bond_number, angle, height, base_r))

    # Save results
    results_file = os.path.join(SCRIPT_DIR, "results.csv")
    with open(results_file, "w") as f:
        f.write("label,gravity,surface_energy,bond_number,angle_deg,pile_height,base_radius\n")
        for label, gz, gamma, bo, angle, height, base_r in results:
            if isinstance(angle, float) and np.isnan(angle):
                f.write(f"{label},{gz},{gamma},{bo:.6f},NaN,NaN,NaN\n")
            else:
                f.write(f"{label},{gz},{gamma},{bo:.6f},{angle:.2f},{height:.6f},{base_r:.6f}\n")

    print(f"\nResults saved to: {results_file}")
    print("\nSummary:")
    print(f"{'Label':<25} {'g':>6} {'gamma':>8} {'Bo':>8} {'Angle':>8} {'Height':>8} {'Base':>8}")
    print("-" * 80)
    for label, gz, gamma, bo, angle, height, base_r in results:
        a_str = f"{angle:.1f}" if not (isinstance(angle, float) and np.isnan(angle)) else "N/A"
        h_str = f"{height*1000:.1f}" if not (isinstance(height, float) and np.isnan(height)) else "N/A"
        b_str = f"{base_r*1000:.1f}" if not (isinstance(base_r, float) and np.isnan(base_r)) else "N/A"
        print(f"{label:<25} {gz:>6.2f} {gamma:>8.4f} {bo:>8.3f} {a_str:>8} {h_str:>8} {b_str:>8}")

    return results


if __name__ == "__main__":
    main()

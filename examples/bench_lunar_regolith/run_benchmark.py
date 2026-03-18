#!/usr/bin/env python3
"""
Run the full lunar regolith cohesive angle of repose benchmark.

Generates TOML configs for multiple (gravity, surface_energy) combinations,
runs each simulation using the draining-box method, measures the pile
profile from the final particle dump, and saves results to results.csv.

Method: Particles settle in a walled box, the "gate" wall is removed,
particles drain out, and the remaining pile's angle is measured.

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
# Method: draining box

[comm]
processors_x = 1
processors_y = 1
processors_z = 1

[domain]
x_low = -0.04
x_high = 0.12
y_low = 0.0
y_high = 0.010
z_low = 0.0
z_high = 0.15
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
restitution = 0.1
friction = 0.5
rolling_friction = 1.0
surface_energy = {surface_energy}

# Count-based insertion in a narrow column (20mm wide, ~105mm tall packed)
[[particles.insert]]
material = "regolith"
radius = 0.001
density = 1500.0
count = 3000
velocity_z = -0.5
region = {{ type = "block", min = [-0.03, 0.0, 0.005], max = [-0.01, 0.010, 0.145] }}

# Floor wall
[[wall]]
point_z = 0.0
normal_z = 1.0
material = "regolith"

# Left wall (containment)
[[wall]]
point_x = -0.04
normal_x = 1.0
material = "regolith"

# Far right wall (catch boundary)
[[wall]]
point_x = 0.12
normal_x = -1.0
material = "regolith"

# Gate wall — removed after settling
[[wall]]
point_x = -0.01
normal_x = -1.0
material = "regolith"
name = "gate"

[output]
dir = "{output_dir}"

# Stage 1: settle particles in the box
[[run]]
name = "settle"
steps = 150000
thermo = 15000
dt = 5e-6

# Stage 2: gate removed, particles drain
[[run]]
name = "drain"
steps = 200000
thermo = 20000
dt = 5e-6

# Stage 3: final settling and measurement
[[run]]
name = "measure"
steps = 100000
thermo = 20000
dt = 5e-6
dump_interval = 100000
save_at_end = true
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
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        print(f"  WARNING: Simulation failed (rc={result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr[-500:]}")
        return False
    return True


def find_last_dump(output_dir):
    """Find the most recently modified dump CSV file in the output directory."""
    pattern = os.path.join(output_dir, "dump", "dump_*_rank0.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time (newest last) to avoid picking old runs
    files.sort(key=os.path.getmtime)
    return files[-1]


def measure_pile_angle(dump_file):
    """
    Measure the pile angle from a draining-box particle dump.

    After the gate is removed, particles drain rightward. The remaining pile
    rests against the left wall with a sloped free surface on the right side.

    Method:
    1. Bin particles by x position (no symmetry — pile is against the left wall)
    2. Surface = 90th percentile of z per bin
    3. Find the sloped region between the flat top and the base
    4. Fit a line to the slope
    5. Angle = atan(|slope|)

    Returns: (angle_deg, pile_height, base_x)
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

    surface_top = z + r  # Top of each particle

    # Only consider particles that are on the floor (not fallen off far right)
    # The box region was x=[-0.05, -0.01], so pile is near the left wall
    # Include some margin for the sloped part
    x_min = np.min(x) - RADIUS
    x_max_pile = np.percentile(x, 95)  # Ignore far-flung outliers

    # Bin by x position
    n_bins = 50
    bin_edges = np.linspace(x_min, x_max_pile, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    surface_z = np.full(n_bins, np.nan)

    for b in range(n_bins):
        mask = (x >= bin_edges[b]) & (x < bin_edges[b + 1])
        if np.sum(mask) >= 3:
            surface_z[b] = np.percentile(surface_top[mask], 90)

    valid = ~np.isnan(surface_z)
    if np.sum(valid) < 3:
        return None, None, None

    xc = bin_centers[valid]
    zc = surface_z[valid]

    # Find peak height (should be near the left wall)
    floor_z = RADIUS
    peak_z = np.max(zc)
    pile_height = peak_z - floor_z

    if pile_height < 3 * RADIUS:
        return None, None, None

    # Fit the sloped region: bins where surface height is between
    # 20% and 80% of peak height (avoids flat top and scattered tail)
    h_low = floor_z + 0.2 * pile_height
    h_high = floor_z + 0.8 * pile_height
    slope_mask = (zc >= h_low) & (zc <= h_high)

    if np.sum(slope_mask) < 2:
        # Fallback: use all bins above floor + 2 radii
        slope_mask = zc > floor_z + 2 * RADIUS
        if np.sum(slope_mask) < 2:
            # Last resort: simple height/base estimate
            base_x = xc[zc > floor_z + RADIUS].max() if np.any(zc > floor_z + RADIUS) else xc[-1]
            top_x = xc[np.argmax(zc)]
            angle_deg = np.degrees(np.arctan2(pile_height, abs(base_x - top_x)))
            return angle_deg, pile_height, base_x

    xfit = xc[slope_mask]
    zfit = zc[slope_mask]

    coeffs = np.polyfit(xfit, zfit, 1)
    slope = coeffs[0]  # Should be negative (height decreases with x)

    angle_deg = np.degrees(np.arctan(abs(slope)))

    # Base x: where the fitted line crosses the floor level
    if abs(slope) > 1e-6:
        base_x = (floor_z - coeffs[1]) / slope
    else:
        base_x = xc[-1]

    return angle_deg, pile_height, base_x


def main():
    print("=" * 60)
    print("Lunar Regolith Cohesive Angle of Repose Benchmark")
    print("Method: Draining box (settle, remove gate, measure pile)")
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

        angle, height, base_x = measure_pile_angle(dump_file)
        if angle is None:
            print("  WARNING: Could not measure pile angle")
            results.append((label, gz, gamma, bond_number, np.nan, np.nan, np.nan))
        else:
            print(f"  Angle: {angle:.1f} deg, Height: {height*1000:.1f} mm, Base x: {base_x*1000:.1f} mm")
            results.append((label, gz, gamma, bond_number, angle, height, base_x))

    # Save results
    results_file = os.path.join(SCRIPT_DIR, "results.csv")
    with open(results_file, "w") as f:
        f.write("label,gravity,surface_energy,bond_number,angle_deg,pile_height,base_x\n")
        for label, gz, gamma, bo, angle, height, base_x in results:
            if isinstance(angle, float) and np.isnan(angle):
                f.write(f"{label},{gz},{gamma},{bo:.6f},NaN,NaN,NaN\n")
            else:
                f.write(f"{label},{gz},{gamma},{bo:.6f},{angle:.2f},{height:.6f},{base_x:.6f}\n")

    print(f"\nResults saved to: {results_file}")
    print("\nSummary:")
    print(f"{'Label':<25} {'g':>6} {'gamma':>8} {'Bo':>8} {'Angle':>8} {'Height':>8} {'Base_x':>8}")
    print("-" * 80)
    for label, gz, gamma, bo, angle, height, base_x in results:
        a_str = f"{angle:.1f}" if not (isinstance(angle, float) and np.isnan(angle)) else "N/A"
        h_str = f"{height*1000:.1f}" if not (isinstance(height, float) and np.isnan(height)) else "N/A"
        b_str = f"{base_x*1000:.1f}" if not (isinstance(base_x, float) and np.isnan(base_x)) else "N/A"
        print(f"{label:<25} {gz:>6.2f} {gamma:>8.4f} {bo:>8.3f} {a_str:>8} {h_str:>8} {b_str:>8}")

    return results


if __name__ == "__main__":
    main()

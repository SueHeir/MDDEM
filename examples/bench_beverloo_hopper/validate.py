#!/usr/bin/env python3
"""
Validate Beverloo hopper discharge benchmark.

Compares measured steady-state mass flow rate from each orifice-width run
against the 2D Beverloo correlation:

    W = C * rho_bulk * sqrt(g) * (D - k*d)^(3/2)

where:
    C ~ 0.58      empirical constant
    k ~ 1.4       empty annulus correction
    d = 0.002 m   particle diameter
    g = 9.81 m/s² gravity
    rho_bulk      bulk density (particle density × packing fraction)

Each run's data file: data/particle_count_D{N}d.txt
Columns: step  time  count  mass

PASS if measured flow rate is within 30% of Beverloo prediction for each orifice.
(DEM with limited particles has statistical noise; 30% is a reasonable tolerance.)
"""

import os
import sys
import glob
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

# Physical parameters
d = 0.002          # particle diameter [m]
g = 9.81           # gravity [m/s²]
rho_particle = 2500.0  # particle density [kg/m³]
phi = 0.60         # approximate 2D packing fraction
rho_bulk = rho_particle * phi  # bulk density [kg/m³]
y_depth = 0.004    # periodic y-extent (slab thickness) [m]

# Beverloo constants
C_beverloo = 0.58
k_beverloo = 1.4

# Tolerance: accept within 30% of prediction
tolerance = 0.30


def beverloo_2d(D, d_part, depth):
    """2D Beverloo mass flow rate: W = C * rho_bulk * sqrt(g) * (D - k*d)^(3/2) * depth."""
    effective_D = D - k_beverloo * d_part
    if effective_D <= 0:
        return 0.0
    return C_beverloo * rho_bulk * np.sqrt(g) * effective_D**1.5 * depth


def measure_flow_rate(filepath):
    """Compute steady-state mass flow rate from particle count data.

    Uses linear regression on the mass vs time curve during the
    steady-state discharge period (middle 60% of data, avoiding
    initial transient and final emptying).
    """
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    time = data[:, 1]
    mass = data[:, 3]

    if len(time) < 5:
        print(f"  WARNING: too few data points ({len(time)}) in {filepath}")
        return None, None, None

    # Use middle 60% of data for steady-state measurement
    n = len(time)
    i_start = max(1, int(0.2 * n))
    i_end = min(n - 1, int(0.8 * n))

    t_ss = time[i_start:i_end]
    m_ss = mass[i_start:i_end]

    if len(t_ss) < 3:
        print(f"  WARNING: too few steady-state points ({len(t_ss)})")
        return None, None, None

    # Linear fit: mass = m0 - W * (t - t0)
    # flow_rate = -slope
    coeffs = np.polyfit(t_ss, m_ss, 1)
    flow_rate = -coeffs[0]  # negative slope → positive flow rate

    return flow_rate, t_ss, m_ss


print("=" * 60)
print("Beverloo Hopper Discharge Benchmark Validation")
print("=" * 60)
print(f"  Particle diameter d = {d*1000:.1f} mm")
print(f"  Bulk density rho_b = {rho_bulk:.0f} kg/m³")
print(f"  Beverloo C = {C_beverloo}, k = {k_beverloo}")
print(f"  Tolerance = ±{tolerance*100:.0f}%")
print()

# Find data files
pattern = os.path.join(data_dir, "particle_count_D*d.txt")
files = sorted(glob.glob(pattern))

if not files:
    # Try output directories directly
    for mult in [5, 8, 12, 16]:
        out_dir = os.path.join(script_dir, f"output_D{mult}d", "data")
        f = os.path.join(out_dir, "particle_count.txt")
        if os.path.isfile(f):
            files.append(f)

if not files:
    print("ERROR: No data files found. Run the simulation first.")
    print(f"  Expected: {pattern}")
    sys.exit(1)

passed = 0
total = 0
results = []

for filepath in files:
    # Parse orifice multiplier from filename or directory
    basename = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
    if "D" in basename and "d" in basename:
        mult_str = basename.split("D")[1].split("d")[0]
    else:
        basename2 = os.path.basename(filepath)
        if "D" in basename2:
            mult_str = basename2.split("D")[1].split("d")[0]
        else:
            print(f"  Skipping {filepath}: cannot parse orifice width")
            continue

    try:
        mult = int(mult_str)
    except ValueError:
        try:
            mult = float(mult_str)
        except ValueError:
            print(f"  Skipping {filepath}: cannot parse multiplier '{mult_str}'")
            continue

    D = mult * d  # orifice width [m]
    W_theory = beverloo_2d(D, d, y_depth)

    flow_rate, t_ss, m_ss = measure_flow_rate(filepath)

    total += 1

    if flow_rate is None:
        print(f"  D = {mult}d ({D*1000:.1f} mm): FAIL (insufficient data)")
        results.append((D, None, W_theory))
        continue

    rel_error = abs(flow_rate - W_theory) / W_theory if W_theory > 0 else float('inf')

    if rel_error <= tolerance:
        status = "PASS"
        passed += 1
    else:
        status = "FAIL"

    print(f"  D = {mult:>2g}d ({D*1000:5.1f} mm): W_meas = {flow_rate:.4e} kg/s, "
          f"W_bev = {W_theory:.4e} kg/s, err = {rel_error*100:5.1f}%  [{status}]")
    results.append((D, flow_rate, W_theory))

print()
print(f"Results: {passed}/{total} orifice widths within ±{tolerance*100:.0f}% of Beverloo")

if passed == total and total > 0:
    print("ALL CHECKS PASSED")
else:
    print(f"WARNING: {total - passed} check(s) failed")

sys.exit(0 if passed == total and total > 0 else 1)

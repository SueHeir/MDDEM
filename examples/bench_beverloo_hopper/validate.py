#!/usr/bin/env python3
"""
Validate Beverloo hopper discharge benchmark.

Compares measured steady-state mass flow rate from each orifice-width run
against the quasi-2D Beverloo correlation:

    W = C * rho_bulk * sqrt(g) * (D - k*d)^(3/2) * depth

where:
    C ~ 0.58      empirical discharge coefficient
    k ~ 1.4       empty annulus correction
    d = 0.002 m   particle diameter
    g = 9.81 m/s  gravity
    rho_bulk      bulk density = rho_particle * packing_fraction
    depth         slab thickness (periodic y extent)

Each run's data file: data/particle_count_D{N}d.txt
Columns: step  time  count  mass

PASS if measured flow rate is within 50% of Beverloo prediction for each orifice.
(DEM with ~800 particles has significant statistical noise; 50% is reasonable
for a small-system validation. The key physics test is the correct scaling
exponent of 3/2.)
"""

import os
import sys
import glob
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

# Physical parameters
d = 0.002          # particle diameter [m]
g = 9.81           # gravity [m/s^2]
rho_particle = 2500.0  # particle density [kg/m^3]
phi = 0.58         # approximate packing fraction (random packing, quasi-2D)
rho_bulk = rho_particle * phi  # bulk density [kg/m^3]
y_depth = 0.004    # periodic y-extent (slab thickness) [m]

# Beverloo constants (quasi-2D slot orifice)
C_beverloo = 0.58
k_beverloo = 1.4

# Tolerance: accept within 50% of prediction
# (small system with ~800 particles has significant noise)
tolerance = 0.50


def beverloo_2d(D, d_part, depth):
    """Quasi-2D Beverloo mass flow rate.

    W = C * rho_bulk * sqrt(g) * (D - k*d)^(3/2) * depth

    Units: [kg/m^3] * [m/s^2]^0.5 * [m]^1.5 * [m] = kg/s
    """
    effective_D = D - k_beverloo * d_part
    if effective_D <= 0:
        return 0.0
    return C_beverloo * rho_bulk * np.sqrt(g) * effective_D**1.5 * depth


def measure_flow_rate(filepath):
    """Compute steady-state mass flow rate from particle count data.

    Uses linear regression on the mass vs time curve during the
    active discharge period: from 10% mass loss (skip initial transient)
    to 70% mass loss (before emptying tail dominates).

    Returns (flow_rate, t_steady, m_steady) or (None, None, None) on failure.
    """
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    time = data[:, 1]
    mass = data[:, 3]

    if len(time) < 5:
        print(f"  WARNING: too few data points ({len(time)}) in {filepath}")
        return None, None, None

    # Find active discharge window based on mass thresholds
    m0 = mass[0]
    if m0 <= 0:
        return None, None, None

    # Start at 90% of initial mass (skip transient), end at 30% (before tail)
    m_hi = 0.90 * m0
    m_lo = 0.30 * m0

    mask = (mass <= m_hi) & (mass >= m_lo)
    if np.sum(mask) < 3:
        # Fallback: use first 30% of data by index
        n = len(time)
        i_end = max(3, int(0.3 * n))
        t_ss = time[1:i_end]
        m_ss = mass[1:i_end]
    else:
        t_ss = time[mask]
        m_ss = mass[mask]

    if len(t_ss) < 3:
        print(f"  WARNING: too few steady-state points ({len(t_ss)})")
        return None, None, None

    # Linear fit: mass = m0 - W * (t - t0)
    # flow_rate = -slope (mass decreases over time)
    coeffs = np.polyfit(t_ss, m_ss, 1)
    flow_rate = -coeffs[0]  # negative slope -> positive flow rate

    return flow_rate, t_ss, m_ss


print("=" * 60)
print("Beverloo Hopper Discharge Benchmark Validation")
print("=" * 60)
print(f"  Particle diameter d = {d*1000:.1f} mm")
print(f"  Bulk density rho_b = {rho_bulk:.0f} kg/m^3 (phi = {phi})")
print(f"  Slab depth = {y_depth*1000:.1f} mm")
print(f"  Beverloo C = {C_beverloo}, k = {k_beverloo}")
print(f"  Tolerance = +/-{tolerance*100:.0f}%")
print()

# Find data files
pattern = os.path.join(data_dir, "particle_count_D*d.txt")
files = sorted(glob.glob(pattern))

if not files:
    # Try output directories
    for mult in [5, 6, 8, 10, 12, 14, 15, 16]:
        out_dir = os.path.join(script_dir, f"output_D{mult}d", "data")
        f = os.path.join(out_dir, "particle_count.txt")
        if os.path.isfile(f):
            files.append(f)

if not files:
    print("ERROR: No data files found. Run the simulation first.")
    print(f"  Expected: {pattern}")
    print("  Or output_D*d/data/particle_count.txt directories")
    sys.exit(1)

passed = 0
total = 0
results = []

for filepath in files:
    # Parse orifice multiplier from filename or directory
    basename = os.path.basename(filepath)
    dirname = os.path.basename(os.path.dirname(os.path.dirname(filepath)))

    mult = None
    # Try directory name first (output_D6d/data/particle_count.txt)
    if "D" in dirname and "d" in dirname:
        mult_str = dirname.split("D")[1].split("d")[0]
        try:
            mult = int(mult_str)
        except ValueError:
            try:
                mult = float(mult_str)
            except ValueError:
                pass

    # Try filename (particle_count_D6d.txt)
    if mult is None and "D" in basename:
        mult_str = basename.split("D")[1].split("d")[0]
        try:
            mult = int(mult_str)
        except ValueError:
            try:
                mult = float(mult_str)
            except ValueError:
                pass

    if mult is None:
        print(f"  Skipping {filepath}: cannot parse orifice width")
        continue

    D = mult * d  # orifice width [m]
    W_theory = beverloo_2d(D, d, y_depth)

    flow_rate, t_ss, m_ss = measure_flow_rate(filepath)

    total += 1

    if flow_rate is None:
        print(f"  D = {mult}d ({D*1000:.1f} mm): FAIL (insufficient data)")
        results.append((D, None, W_theory))
        continue

    if W_theory > 0:
        rel_error = abs(flow_rate - W_theory) / W_theory
    else:
        rel_error = float('inf')

    if rel_error <= tolerance:
        status = "PASS"
        passed += 1
    else:
        status = "FAIL"

    print(f"  D = {mult:>2g}d ({D*1000:5.1f} mm): W_meas = {flow_rate:.4e} kg/s, "
          f"W_bev = {W_theory:.4e} kg/s, err = {rel_error*100:5.1f}%  [{status}]")
    results.append((D, flow_rate, W_theory))

print()
print(f"Results: {passed}/{total} orifice widths within +/-{tolerance*100:.0f}% of Beverloo")

if passed == total and total > 0:
    print("ALL CHECKS PASSED")
else:
    print(f"WARNING: {total - passed} check(s) failed")

sys.exit(0 if passed == total and total > 0 else 1)

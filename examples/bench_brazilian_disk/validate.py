#!/usr/bin/env python3
"""
Validate Brazilian disk tensile test benchmark.

Checks on data/load_displacement.csv:
  1. Load increases (elastic loading phase exists)
  2. Peak load is followed by load drop (brittle failure)
  3. Bonds break during the test
  4. Tensile strength is in expected range for bond parameters
  5. No NaN/Inf values
"""

import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, "data", "load_displacement.csv")

if not os.path.isfile(data_file):
    print(f"ERROR: {data_file} not found. Run simulation first.")
    sys.exit(1)

data = np.genfromtxt(data_file, delimiter=",", skip_header=1, dtype=float)
if data.ndim == 1:
    data = data.reshape(1, -1)

# Columns: step, time, load, gap, z_bottom, z_top, f_bottom, f_top, bonds_broken, bond_count
steps = data[:, 0]
time = data[:, 1]
load = data[:, 2]
gap = data[:, 3]
bonds_broken = data[:, 8]
bond_count = data[:, 9]

print("=" * 55)
print("Brazilian Disk Tensile Test Validation")
print("=" * 55)

passed = 0
total = 0

# 1. No NaN/Inf
total += 1
if np.all(np.isfinite(load)) and np.all(np.isfinite(gap)):
    print("  No NaN/Inf:              PASS")
    passed += 1
else:
    print("  No NaN/Inf:              FAIL")

# 2. Load increases (elastic loading phase)
total += 1
peak_load = np.max(load)
if peak_load > 0:
    print(f"  Load buildup:            PASS (peak = {peak_load:.4e} N)")
    passed += 1
else:
    print(f"  Load buildup:            FAIL (peak = {peak_load:.4e} N, expected > 0)")

# 3. Peak load followed by drop (brittle failure)
total += 1
if peak_load > 0:
    peak_idx = np.argmax(load)
    if peak_idx < len(load) - 1:
        post_peak_min = np.min(load[peak_idx:])
        drop_ratio = post_peak_min / peak_load if peak_load > 0 else 1.0
        if drop_ratio < 0.5:
            print(f"  Brittle failure:         PASS (post-peak drop to {drop_ratio:.1%} of peak)")
            passed += 1
        else:
            print(f"  Brittle failure:         FAIL (post-peak ratio = {drop_ratio:.1%}, expected < 50%)")
    else:
        print("  Brittle failure:         FAIL (peak at last step, no post-peak data)")
else:
    print("  Brittle failure:         SKIP (no load)")

# 4. Bonds break during test
total += 1
max_broken = np.max(bonds_broken)
if max_broken > 0:
    print(f"  Bonds broken:            PASS ({int(max_broken)} bonds broken)")
    passed += 1
else:
    print("  Bonds broken:            FAIL (no bonds broke)")

# 5. Tensile strength estimate
# Brazilian test: sigma_t = 2*P / (pi * D * t)
# D = 0.02 m (diameter), t = 0.001 m (thickness ~ particle diameter in y)
total += 1
D = 0.02  # disk diameter [m]
t = 0.001  # thickness (one particle diameter) [m]
if peak_load > 0:
    sigma_t = 2 * peak_load / (np.pi * D * t)
    # Expected: bond_stiffness * break_strain * (bond_area / contact_area)
    # With k_n = 1e8, break = 0.005, r = 0.0005:
    # Bond force at break ~ k_n * 0.005 * 2*r = 1e8 * 0.005 * 0.001 = 500 N
    # Very approximate: sigma_t should be order of 1e4 to 1e7 Pa
    if 1e2 < sigma_t < 1e8:
        print(f"  Tensile strength:        PASS (sigma_t = {sigma_t:.2e} Pa)")
        passed += 1
    else:
        print(f"  Tensile strength:        FAIL (sigma_t = {sigma_t:.2e} Pa, expected 1e2-1e8 Pa)")
else:
    print("  Tensile strength:        SKIP (no load)")

print(f"\nResults: {passed}/{total} checks passed")
if passed == total:
    print("ALL CHECKS PASSED")
else:
    print(f"WARNING: {total - passed} check(s) failed")

sys.exit(0 if passed == total else 1)

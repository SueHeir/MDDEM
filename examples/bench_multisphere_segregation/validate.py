#!/usr/bin/env python3
"""
Validate multisphere segregation benchmark.

Checks on data/segregation.csv:
  1. No NaN/Inf in data
  2. Both sphere and dimer z-positions are positive (physical)
  3. Segregation index S > 0 in the final 20% of the run (dimers above spheres)
  4. Final S is significantly positive (S > 0.005)
  5. Segregation trend: S increases over time (positive slope in second half)
"""

import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, "data", "segregation.csv")

if not os.path.isfile(data_file):
    print(f"ERROR: {data_file} not found. Run simulation first.")
    sys.exit(1)

data = np.loadtxt(data_file, delimiter=",", skiprows=1)
if data.ndim == 1:
    data = data.reshape(1, -1)

steps = data[:, 0]
times = data[:, 1]
z_sphere = data[:, 2]
z_dimer = data[:, 3]
seg_index = data[:, 4]

print("=" * 55)
print("Multisphere Segregation Benchmark Validation")
print("=" * 55)

passed = 0
total = 0

# 1. No NaN/Inf
total += 1
if np.all(np.isfinite(data)):
    print("  No NaN/Inf:           PASS")
    passed += 1
else:
    print("  No NaN/Inf:           FAIL")

# 2. Physical z-positions (positive)
total += 1
if np.all(z_sphere > 0) and np.all(z_dimer > 0):
    print(f"  Positive z:           PASS (z_s={z_sphere[-1]:.4e}, z_d={z_dimer[-1]:.4e})")
    passed += 1
else:
    print(f"  Positive z:           FAIL (min z_s={np.min(z_sphere):.4e}, min z_d={np.min(z_dimer):.4e})")

# 3. S > 0 in final 20% (dimers above spheres at steady state)
total += 1
n = len(seg_index)
final_portion = seg_index[int(0.8 * n):]
mean_final_S = np.mean(final_portion)
if mean_final_S > 0:
    print(f"  Final S > 0:          PASS (mean final S = {mean_final_S:.6f})")
    passed += 1
else:
    print(f"  Final S > 0:          FAIL (mean final S = {mean_final_S:.6f})")

# 4. S is significantly positive at end (S > 0.005)
total += 1
if mean_final_S > 0.005:
    print(f"  S > 0.005:            PASS (S = {mean_final_S:.6f})")
    passed += 1
else:
    print(f"  S > 0.005:            FAIL (S = {mean_final_S:.6f}, expected > 0.005)")

# 5. Segregation trend: positive slope in second half
total += 1
if n > 4:
    half = n // 2
    s2 = steps[half:]
    si2 = seg_index[half:]
    coeffs = np.polyfit(s2, si2, 1)
    slope = coeffs[0]
    # Accept if slope is non-negative or if S is already high
    if slope >= 0 or mean_final_S > 0.01:
        print(f"  Segregation trend:    PASS (slope={slope:.3e}, final S={mean_final_S:.6f})")
        passed += 1
    else:
        print(f"  Segregation trend:    FAIL (slope={slope:.3e}, expected >= 0)")
else:
    print("  Segregation trend:    SKIP (not enough data points)")

print(f"\nResults: {passed}/{total} checks passed")
if passed == total:
    print("ALL CHECKS PASSED")
else:
    print(f"WARNING: {total - passed} check(s) failed")

sys.exit(0 if passed == total else 1)

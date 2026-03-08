#!/usr/bin/env python3
"""
Validate hopper simulation: gravity-dominated settling with walls.

Checks on data/GranularTemp.txt (columns: step, time, temperature):
  1. No NaN/Inf in temperature
  2. All temperatures >= 0
  3. No energy explosion (max T bounded)
"""

import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, "data", "GranularTemp.txt")

if not os.path.isfile(data_file):
    print(f"ERROR: {data_file} not found. Run simulation first.")
    sys.exit(1)

data = np.loadtxt(data_file)
if data.ndim == 1:
    data = data.reshape(1, -1)

temps = data[:, 2]  # column 2 is temperature

print("=" * 50)
print("Hopper Validation")
print("=" * 50)

passed = 0
total = 0

# 1. No NaN/Inf
total += 1
if np.all(np.isfinite(temps)):
    print("  No NaN/Inf:        PASS")
    passed += 1
else:
    print("  No NaN/Inf:        FAIL")

# 2. All temperatures >= 0
total += 1
if np.all(temps >= 0):
    print("  Non-negative T:    PASS")
    passed += 1
else:
    print(f"  Non-negative T:    FAIL (min={np.min(temps):.6e})")

# 3. No energy explosion (hopper temps are very small, ~1e-29 to 1e-31)
# Just check max is bounded reasonably (< 1.0 would be absurd for this system)
total += 1
T_max = np.max(temps)
if T_max < 1.0:
    print(f"  Bounded T:         PASS (max={T_max:.6e})")
    passed += 1
else:
    print(f"  Bounded T:         FAIL (max={T_max:.6e}, expected < 1.0)")

print(f"\nResults: {passed}/{total} checks passed")
if passed == total:
    print("ALL CHECKS PASSED")
else:
    print(f"WARNING: {total - passed} check(s) failed")

sys.exit(0 if passed == total else 1)

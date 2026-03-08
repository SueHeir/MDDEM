#!/usr/bin/env python3
"""
Validate granular gas benchmark: inelastic spheres must cool (Haff's law).

Checks on data/GranularTemp.txt (columns: step, time, temperature):
  1. No NaN/Inf in temperature
  2. All temperatures >= 0
  3. Final T < initial T (must cool with e=0.95)
  4. No energy explosion: max(T) < 2x initial T
  5. Cooling trend: linear-fit slope of T vs step is negative
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

steps = data[:, 0]
temps = data[:, 2]  # column 2 is temperature

print("=" * 50)
print("Granular Gas Benchmark Validation")
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

# 3. Final T < initial T (cooling)
total += 1
T_init = temps[0]
T_final = temps[-1]
if T_init > 0 and T_final < T_init:
    print(f"  Cooling (Tf<Ti):   PASS (Ti={T_init:.6e}, Tf={T_final:.6e})")
    passed += 1
else:
    print(f"  Cooling (Tf<Ti):   FAIL (Ti={T_init:.6e}, Tf={T_final:.6e})")

# 4. No energy explosion
total += 1
T_max = np.max(temps)
if T_init > 0 and T_max < 2.0 * T_init:
    print(f"  No explosion:      PASS (max={T_max:.6e} < 2*Ti={2*T_init:.6e})")
    passed += 1
else:
    print(f"  No explosion:      FAIL (max={T_max:.6e}, 2*Ti={2*T_init:.6e})")

# 5. Cooling trend (negative slope, skip step 0)
total += 1
if len(steps) > 2:
    s = steps[1:]
    t = temps[1:]
    coeffs = np.polyfit(s, t, 1)
    slope = coeffs[0]
    if slope < 0:
        print(f"  Cooling slope:     PASS (slope={slope:.6e})")
        passed += 1
    else:
        print(f"  Cooling slope:     FAIL (slope={slope:.6e}, expected < 0)")
else:
    print("  Cooling slope:     SKIP (not enough data points)")

print(f"\nResults: {passed}/{total} checks passed")
if passed == total:
    print("ALL CHECKS PASSED")
else:
    print(f"WARNING: {total - passed} check(s) failed")

sys.exit(0 if passed == total else 1)

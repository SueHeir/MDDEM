#!/usr/bin/env python3
"""Validate lj_langevin example output: check temperature stabilizes near target."""

import sys
import re

target_temp = 0.85
tolerance = 0.15  # allow ±0.15 for stochastic thermostat

temps = []
for line in sys.stdin:
    # Match thermo output lines: first column is step number
    parts = line.split()
    if len(parts) >= 3 and parts[0].isdigit():
        step = int(parts[0])
        # Look for Temp column (typically 3rd column)
        try:
            temp = float(parts[2])
            if step > 2000:  # skip equilibration
                temps.append(temp)
        except (ValueError, IndexError):
            continue

if not temps:
    print("ERROR: No temperature data found in thermo output")
    sys.exit(1)

avg_temp = sum(temps) / len(temps)
print(f"Average temperature (after equilibration): {avg_temp:.4f}")
print(f"Target: {target_temp}, Tolerance: ±{tolerance}")

if abs(avg_temp - target_temp) < tolerance:
    print("PASS: Temperature within tolerance")
else:
    print(f"FAIL: Temperature {avg_temp:.4f} outside tolerance of {target_temp} ± {tolerance}")
    sys.exit(1)

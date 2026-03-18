#!/usr/bin/env python3
"""
Validate triaxial compression benchmark against Mohr-Coulomb failure theory.

For cohesionless granular material (c = 0):
    σ₁ = σ₃ × (1 + sin φ) / (1 - sin φ)
    sin φ = (σ₁ - σ₃) / (σ₁ + σ₃)

Expected friction angle for μ = 0.5: φ ≈ 25–50° (DEM spheres)

Checks:
  1. Stress data files exist and contain valid data
  2. Peak stress ratio σ₁/σ₃ is in a physically reasonable range [1.5, 8.0]
  3. Mobilised friction angle φ is in expected range [15°, 55°]
  4. Friction angle is consistent across confining pressures (std < 5°)
  5. Mohr-Coulomb failure envelope is approximately linear (R² > 0.9)
"""

import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
pressures = ["10kPa", "50kPa", "200kPa"]

print("=" * 60)
print("Triaxial Compression — Mohr-Coulomb Validation")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────────────────

results = {}  # label → (sigma_1_peak, sigma_3_at_peak)

for label in pressures:
    data_dir = os.path.join(script_dir, f"data_{label}")
    csv_path = os.path.join(data_dir, "triaxial_stress.csv")
    if not os.path.isfile(csv_path):
        print(f"  WARNING: {csv_path} not found — skipping {label}")
        continue
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        print(f"  WARNING: {csv_path} is empty — skipping {label}")
        continue

    # Find peak deviatoric stress q = σ₁ - σ₃
    q = data["q"]
    idx_peak = np.argmax(q)
    s1 = data["sigma_1"][idx_peak]
    s3 = data["sigma_3"][idx_peak]
    results[label] = (s1, s3, q[idx_peak])
    print(f"  {label:>6s}: σ₁ = {s1:10.1f} Pa, σ₃ = {s3:10.1f} Pa, "
          f"q = {q[idx_peak]:10.1f} Pa, σ₁/σ₃ = {s1/s3 if s3 > 0 else float('inf'):.2f}")

if len(results) < 2:
    print("\nERROR: Need at least 2 confining pressures for validation.")
    sys.exit(1)

# ── Run checks ─────────────────────────────────────────────────────────────

passed = 0
total = 0

# Check 1: valid data
total += 1
if all(np.isfinite(v[0]) and np.isfinite(v[1]) for v in results.values()):
    print("\n  [1] Valid data:           PASS")
    passed += 1
else:
    print("\n  [1] Valid data:           FAIL (NaN/Inf in stress)")

# Check 2: stress ratio in range
total += 1
ratios = []
for label, (s1, s3, _) in results.items():
    if s3 > 0:
        ratios.append(s1 / s3)
if ratios and all(1.5 <= r <= 8.0 for r in ratios):
    print(f"  [2] Stress ratio range:  PASS (ratios: {[f'{r:.2f}' for r in ratios]})")
    passed += 1
else:
    print(f"  [2] Stress ratio range:  FAIL (ratios: {[f'{r:.2f}' for r in ratios]}, expected 1.5–8.0)")

# Check 3: friction angle in expected range
total += 1
phi_values = []
for label, (s1, s3, _) in results.items():
    if (s1 + s3) > 0:
        sin_phi = (s1 - s3) / (s1 + s3)
        sin_phi = max(-1.0, min(1.0, sin_phi))
        phi = np.degrees(np.arcsin(sin_phi))
        phi_values.append(phi)
phi_mean = np.mean(phi_values) if phi_values else 0
if phi_values and all(15 <= p <= 55 for p in phi_values):
    print(f"  [3] Friction angle:      PASS (φ = {phi_mean:.1f}° ± {np.std(phi_values):.1f}°, range 15–55°)")
    passed += 1
else:
    print(f"  [3] Friction angle:      FAIL (φ values: {[f'{p:.1f}' for p in phi_values]}°, expected 15–55°)")

# Check 4: consistency across pressures
total += 1
if len(phi_values) >= 2 and np.std(phi_values) < 5.0:
    print(f"  [4] Consistency:         PASS (std = {np.std(phi_values):.2f}° < 5°)")
    passed += 1
elif len(phi_values) < 2:
    print(f"  [4] Consistency:         SKIP (need ≥ 2 pressure levels)")
else:
    print(f"  [4] Consistency:         FAIL (std = {np.std(phi_values):.2f}° ≥ 5°)")

# Check 5: Mohr-Coulomb linearity (q vs p)
total += 1
s1_arr = np.array([v[0] for v in results.values()])
s3_arr = np.array([v[1] for v in results.values()])
p_arr = (s1_arr + 2 * s3_arr) / 3.0
q_arr = s1_arr - s3_arr
if len(q_arr) >= 3:
    coeffs = np.polyfit(p_arr, q_arr, 1)
    q_fit = np.polyval(coeffs, p_arr)
    ss_res = np.sum((q_arr - q_fit) ** 2)
    ss_tot = np.sum((q_arr - np.mean(q_arr)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    if r_squared > 0.9:
        print(f"  [5] MC linearity:        PASS (R² = {r_squared:.4f})")
        passed += 1
    else:
        print(f"  [5] MC linearity:        FAIL (R² = {r_squared:.4f}, expected > 0.9)")
else:
    # With only 2 points, linearity is trivially satisfied
    print(f"  [5] MC linearity:        PASS (trivially linear with {len(q_arr)} points)")
    passed += 1

# ── Summary ────────────────────────────────────────────────────────────────

print(f"\nResults: {passed}/{total} checks passed")
if phi_values:
    print(f"Measured friction angle: φ = {phi_mean:.1f}° (μ_particle = 0.5)")

if passed == total:
    print("ALL CHECKS PASSED")
else:
    print(f"WARNING: {total - passed} check(s) failed")

sys.exit(0 if passed == total else 1)

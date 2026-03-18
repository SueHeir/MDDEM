#!/usr/bin/env python3
"""
Validate packed bed thermal conductivity benchmark.

Checks:
  1. Temperature profile is approximately linear at steady state (R² > 0.7)
  2. Wall heat flux has converged (< 20% variation in last 20% of data)
  3. Effective k_eff is physically reasonable (0.001 < k_eff/k_s < 1.0)
  4. All particle temperatures between T_cold and T_hot (with margin)
  5. Average temperature near midpoint (T_hot + T_cold) / 2
"""

import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── Load data ───────────────────────────────────────────────────────────────

profile_file = os.path.join(script_dir, "data", "ThermalProfile.csv")
flux_file = os.path.join(script_dir, "data", "WallHeatFlux.txt")

for f in [profile_file, flux_file]:
    if not os.path.isfile(f):
        print(f"ERROR: {f} not found. Run simulation first.")
        sys.exit(1)

# Temperature profile: z, temperature
profile = np.loadtxt(profile_file, delimiter=",", skiprows=1)
z = profile[:, 0]
T = profile[:, 1]

# Wall heat flux: step, time, Q_wall, T_avg, bottom_z, top_z
flux_data = np.loadtxt(flux_file, comments="#")
if flux_data.ndim == 1:
    flux_data = flux_data.reshape(1, -1)
flux_Q = flux_data[:, 2]
flux_Tavg = flux_data[:, 3]
bottom_z = flux_data[-1, 4]
top_z = flux_data[-1, 5]

# ── Parameters ──────────────────────────────────────────────────────────────

T_hot = 400.0
T_cold = 300.0
delta_T = T_hot - T_cold
R = 0.001           # Particle radius [m]
k_s = 50.0          # Solid thermal conductivity [W/(m·K)]
L_x = 0.012         # Domain width [m]
L_y = 0.012         # Domain width [m]
A = L_x * L_y       # Cross-sectional area [m²]
L_bed = top_z - bottom_z  # Distance between walls [m]

N = len(z)
V_particle = N * (4.0 / 3.0) * np.pi * R**3
V_bed = A * L_bed if L_bed > 0 else 1.0
porosity = 1.0 - V_particle / V_bed

print("=" * 60)
print("Packed Bed Thermal Conductivity Benchmark Validation")
print("=" * 60)
print(f"  Particles:    {N}")
print(f"  Bed height:   {L_bed*1000:.2f} mm")
print(f"  Wall gap:     bottom={bottom_z*1000:.2f} mm, top={top_z*1000:.2f} mm")
print(f"  Porosity:     {porosity:.3f}")
print(f"  T_hot={T_hot}K, T_cold={T_cold}K")
print()

passed = 0
total = 0

# ── Check 1: Linear temperature profile ────────────────────────────────────
total += 1
coeffs = np.polyfit(z, T, 1)
T_pred = np.polyval(coeffs, z)
ss_res = np.sum((T - T_pred) ** 2)
ss_tot = np.sum((T - np.mean(T)) ** 2)
R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

if R2 > 0.7:
    print(f"  Linear profile (R²>0.7): PASS (R²={R2:.4f})")
    passed += 1
else:
    print(f"  Linear profile (R²>0.7): FAIL (R²={R2:.4f})")

# ── Check 2: Steady-state convergence ──────────────────────────────────────
total += 1
n_flux = len(flux_Q)
if n_flux >= 10:
    tail_start = int(0.8 * n_flux)
    Q_tail = flux_Q[tail_start:]
    Q_mean = np.mean(Q_tail)
    Q_std = np.std(Q_tail)
    rel_var = abs(Q_std / Q_mean) if abs(Q_mean) > 1e-30 else float("inf")
    if rel_var < 0.20:
        print(f"  Steady-state (var<20%):  PASS (rel_std={rel_var:.4f})")
        passed += 1
    else:
        print(f"  Steady-state (var<20%):  FAIL (rel_std={rel_var:.4f})")
else:
    print("  Steady-state:            SKIP (insufficient data)")

# ── Check 3: k_eff physically reasonable ───────────────────────────────────
total += 1
Q_ss = np.mean(flux_Q[int(0.8 * n_flux):]) if n_flux >= 10 else flux_Q[-1]
k_eff = abs(Q_ss) * L_bed / (A * delta_T) if L_bed > 0 and A > 0 else 0.0
k_ratio = k_eff / k_s if k_s > 0 else 0.0

if 0.001 < k_ratio < 1.0:
    print(f"  k_eff reasonable:        PASS (k_eff={k_eff:.4f} W/(m·K), k_eff/k_s={k_ratio:.4f})")
    passed += 1
else:
    print(f"  k_eff reasonable:        FAIL (k_eff={k_eff:.4f} W/(m·K), k_eff/k_s={k_ratio:.4f})")

# ── Check 4: Temperature bounds ────────────────────────────────────────────
total += 1
margin = 5.0  # K tolerance
if np.min(T) >= (T_cold - margin) and np.max(T) <= (T_hot + margin):
    print(f"  T in bounds:             PASS (min={np.min(T):.2f}, max={np.max(T):.2f})")
    passed += 1
else:
    print(f"  T in bounds:             FAIL (min={np.min(T):.2f}, max={np.max(T):.2f})")

# ── Check 5: Average temperature near midpoint ────────────────────────────
total += 1
T_mid = (T_hot + T_cold) / 2.0
if abs(np.mean(T) - T_mid) < 25.0:
    print(f"  T_avg near midpoint:     PASS (T_avg={np.mean(T):.2f}, T_mid={T_mid:.1f})")
    passed += 1
else:
    print(f"  T_avg near midpoint:     FAIL (T_avg={np.mean(T):.2f}, T_mid={T_mid:.1f})")

# ── Summary ─────────────────────────────────────────────────────────────────
print()
print(f"  k_eff = {k_eff:.4f} W/(m·K), k_eff/k_s = {k_ratio:.4f}")
print(f"  Steady-state Q = {Q_ss:.4e} W")
print(f"  dT/dz slope = {coeffs[0]:.2f} K/m")
print()
print(f"Results: {passed}/{total} checks passed")
if passed == total:
    print("ALL CHECKS PASSED")
else:
    print(f"WARNING: {total - passed} check(s) failed")

sys.exit(0 if passed == total else 1)

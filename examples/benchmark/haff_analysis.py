"""
Haff's Cooling Law Validation for MDDEM Granular Gas Benchmark
==============================================================
Reads data/GranularTemp.txt produced by MDDEM and compares granular
temperature decay against:
  1. Haff's law (inelastic hard spheres, constant e):
         T(t) = T0 / (1 + t/tc)^2
  2. Viscoelastic Hertz reference slope:
         T ~ t^(-5/3)  [Brilliantov & Pöschel 1996]

The analytical cooling time tc is also predicted from kinetic theory and
compared against the fitted value.

Usage:
    cd data/
    python haff_analysis.py [GranularTemp.txt]

File format expected: three columns — step  time(s)  temperature(m^2/s^2)
(produced by MDDEM with the updated print_granular_temperature system)
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Load data ────────────────────────────────────────────────────────────────

data_file = sys.argv[1] if len(sys.argv) > 1 else "default/data/GranularTemp.txt"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Cannot find {data_file}. Run from the data/ directory.")

raw = np.loadtxt(data_file)

if raw.ndim == 1:
    # Old single-column format — reconstruct time from known parameters
    print("WARNING: single-column file detected (old format). Reconstructing time.")
    print("  Assuming dt=8.36e-7 s, output every 2000 steps.")
    T_arr = raw
    steps = np.arange(len(T_arr)) * 2000
    times = steps * 8.36e-7
else:
    steps = raw[:, 0]
    times = raw[:, 1]
    T_arr = raw[:, 2]

# Drop any zero-temperature rows (e.g. initial output before collisions)
mask = T_arr > 0
steps, times, T_arr = steps[mask], times[mask], T_arr[mask]

T0_data = T_arr[0]
t0_data = times[0]
t = times - t0_data   # time relative to first data point

print(f"Loaded {len(T_arr)} data points")
print(f"T0  = {T0_data:.4e} m²/s²")
print(f"t_span = {t[-1]:.4e} s  ({steps[-1]:.0f} steps)")

# ── Haff's law fit  T(t) = T0 / (1 + t/tc)^2 ────────────────────────────────

def haffs_law(t, T0, tc):
    return T0 / (1.0 + t / tc) ** 2

# Initial guess: tc ~ middle of time range
p0 = [T0_data, t[len(t) // 2] if t[-1] > 0 else 1.0]
try:
    popt, pcov = curve_fit(haffs_law, t, T_arr, p0=p0, maxfev=20000,
                           bounds=([0, 1e-10], [np.inf, np.inf]))
    T0_fit, tc_fit = popt
    perr = np.sqrt(np.diag(pcov))
    fit_ok = True
except RuntimeError:
    print("WARNING: Haff's law curve fit did not converge.")
    T0_fit, tc_fit = T0_data, t[len(t) // 2]
    fit_ok = False

print(f"\nHaff fit:  T0 = {T0_fit:.4e} ± {perr[0]:.1e}  "
      f"tc = {tc_fit:.4e} ± {perr[1]:.1e} s")

# ── Analytical prediction of tc ──────────────────────────────────────────────
# From Haff (1983) / van Noije & Ernst (1998), the cooling equation is:
#   dT/dt = -ω0 * T^(3/2)
# with  ω0 = (4/3) * n * σ² * g0 * sqrt(2π/m) * (1-e²)
# giving  tc = 2 / (ω0 * sqrt(T0))
#
# Parameters — edit to match your input_benchmark if changed
N        = 500          # number of particles
L        = 0.025        # box side length (m)
r        = 0.001        # particle radius (m)
rho_p    = 2500         # particle density (kg/m³)
e        = 0.95         # restitution coefficient (match 'dampening' command)

d        = 2 * r
V        = L ** 3
n        = N / V                                      # number density
phi      = n * (4.0 / 3.0) * math.pi * r ** 3        # volume fraction
m_p      = rho_p * (4.0 / 3.0) * math.pi * r ** 3   # particle mass (kg)

# Carnahan–Starling pair correlation at contact
g0 = (1.0 - phi / 2.0) / (1.0 - phi) ** 3

# ω₀ for the cooling equation dT/dt = -ω₀ T^(3/2) with T in m²/s²:
# collision rate ω_coll = 4·n·d²·g₀·√(πT), energy loss ΔT/T = -(1-e²)/3 per collision
# → ω₀ = (4/3)·n·d²·g₀·(1-e²)·√π  (no mass term — T is specific, not in Joules)
omega0 = (4.0 / 3.0) * n * d ** 2 * g0 * math.sqrt(math.pi) * (1.0 - e ** 2)
tc_theory = 2.0 / (omega0 * math.sqrt(T0_data))

print(f"\nSystem:    n={n:.3e} m⁻³, φ={phi:.3f}, g0={g0:.3f}, m={m_p:.3e} kg")
print(f"Theory:    ω0={omega0:.3e} s⁻¹(m²/s²)^(-1/2)")
print(f"           tc_theory = {tc_theory:.4e} s")
print(f"           tc_fit    = {tc_fit:.4e} s")
print(f"           ratio fit/theory = {tc_fit/tc_theory:.3f}")

# ── Log-log slope of late-time data ──────────────────────────────────────────
# Use the second half of data (past initial transient)
if np.any(t > 0):
    mask_pos = t > 0
    t_pos = t[mask_pos]
    T_pos = T_arr[mask_pos]
    late = slice(len(t_pos) // 2, None)
    if len(t_pos[late]) >= 2:
        slope_meas, _ = np.polyfit(np.log(t_pos[late]), np.log(T_pos[late]), 1)
        print(f"\nMeasured late-time log-log slope: {slope_meas:.3f}")
        print(f"  Haff (inelastic hard sphere):    -2.000")
        print(f"  Viscoelastic Hertz:              -1.667")

# ── Plotting ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 1, figsize=(7, 5))
fig.suptitle("Granular Gas Cooling — Haff's Law Validation", fontsize=13)

t_fine = np.linspace(t[0], t[-1], 2000)
T_fit_curve = haffs_law(t_fine, T0_fit, tc_fit)

# ── Panel 1: linear scale ────────────────────────────────────────────────────
# ax = axes[0]
# ax.scatter(times, T_arr, s=4, color="steelblue", label="MDDEM data", zorder=5)
# if fit_ok:
#     ax.plot(t_fine + t0_data, T_fit_curve, "r-", linewidth=2,
#             label=f"Haff fit  $t_c$={tc_fit:.3e} s")
#     ax.plot(t_fine + t0_data, haffs_law(t_fine, T0_data, tc_theory),
#             "g--", linewidth=1.5, label=f"Haff theory  $t_c$={tc_theory:.3e} s")
# ax.set_xlabel("Physical time  (s)")
# ax.set_ylabel("Granular temperature  (m²/s²)")
# ax.set_title("Linear scale")
# ax.legend(fontsize=8)
# ax.grid(True, alpha=0.3)

# ── Panel 2: log-log scale ───────────────────────────────────────────────────
ax = axes

mask_pos = t > 0
t_pos = t[mask_pos]
T_pos = T_arr[mask_pos]

# Normalize to first positive-time point for cleaner slope comparison
T_norm = T_pos / T_pos[0]
t_norm = t_pos / t_pos[0]

ax.loglog(t_norm, T_norm, "o", markersize=3, color="steelblue",
          label="MDDEM data  (normalised)", zorder=5)

# Fit curve (normalised)
if fit_ok:
    t_fine_pos = np.linspace(t_pos[0], t_pos[-1], 2000)
    T_fit_norm = haffs_law(t_fine_pos, T0_fit, tc_fit) / T_pos[0]
    ax.loglog(t_fine_pos / t_pos[0], T_fit_norm, "r-", linewidth=2,
              label=f"Haff fit  ($t_c$={tc_fit:.2e} s)")

# Reference slopes anchored at the midpoint of the data
t_ref = np.array([t_norm[len(t_norm)//2], t_norm[-1]])
T_mid = T_norm[len(T_norm) // 2]

for exp, color, lbl in [(-2.0, "red", "slope −2  (Haff)"),
                         (-5.0/3.0, "green", "slope −5/3  (viscoelastic)")]:
    T_ref_line = T_mid * (t_ref / t_ref[0]) ** exp
    ax.loglog(t_ref, T_ref_line, "--", color=color, alpha=0.75, linewidth=1.5,
              label=lbl)

ax.set_xlabel("Normalised time  $t / t_1$")
ax.set_ylabel("Normalised temperature  $T / T_1$")
ax.set_title("Log-log scale")
ax.legend(fontsize=8)
# ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
out = "haff_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
plt.show()

#!/usr/bin/env python3
"""
Generate publication-quality plots comparing MDDEM hopper discharge
to the quasi-2D Beverloo correlation.

Produces:
  1. beverloo_comparison.png -- W vs (D - k*d) on log-log axes
  2. mass_vs_time.png -- Mass remaining vs time for each orifice width

Usage:
    python3 examples/bench_beverloo_hopper/plot.py
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

# Physical parameters
d = 0.002            # particle diameter [m]
g = 9.81             # gravity [m/s^2]
rho_particle = 2500.0
phi = 0.58           # packing fraction
rho_bulk = rho_particle * phi
y_depth = 0.004      # slab thickness [m]

C_beverloo = 0.58
k_beverloo = 1.4


def beverloo_2d(D_arr, d_part, depth):
    """Quasi-2D Beverloo: W = C * rho_bulk * sqrt(g) * (D - k*d)^(3/2) * depth."""
    eff = D_arr - k_beverloo * d_part
    eff = np.maximum(eff, 0)
    return C_beverloo * rho_bulk * np.sqrt(g) * eff**1.5 * depth


def measure_flow_rate(filepath):
    """Compute steady-state mass flow rate from particle count data."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    time = data[:, 1]
    mass = data[:, 3]
    if len(time) < 5:
        return None, time, mass

    n = len(time)
    i_start = max(1, int(0.2 * n))
    i_end = min(n - 1, int(0.8 * n))
    t_ss = time[i_start:i_end]
    m_ss = mass[i_start:i_end]
    if len(t_ss) < 3:
        return None, time, mass

    coeffs = np.polyfit(t_ss, m_ss, 1)
    return -coeffs[0], time, mass


def find_data_files():
    """Find particle count data files from sweep runs."""
    files = {}
    # Check data/ directory first
    for f in sorted(glob.glob(os.path.join(data_dir, "particle_count_D*d.txt"))):
        basename = os.path.basename(f)
        mult_str = basename.split("D")[1].split("d")[0]
        try:
            mult = int(mult_str)
        except ValueError:
            mult = float(mult_str)
        files[mult] = f

    # Check output directories
    if not files:
        for mult in [5, 6, 8, 10, 12, 14, 15, 16]:
            out_dir = os.path.join(script_dir, f"output_D{mult}d", "data")
            f = os.path.join(out_dir, "particle_count.txt")
            if os.path.isfile(f):
                files[mult] = f

    return files


data_files = find_data_files()
if not data_files:
    print("No data files found. Run the simulation sweep first.")
    sys.exit(1)

# Measure flow rates
D_vals = []
W_meas = []
mult_labels = []
time_data = {}

for mult in sorted(data_files.keys()):
    D = mult * d
    W, t, m = measure_flow_rate(data_files[mult])
    if W is not None and W > 0:
        D_vals.append(D)
        W_meas.append(W)
        mult_labels.append(mult)
    time_data[mult] = (t, m)

D_vals = np.array(D_vals)
W_meas = np.array(W_meas)

# -- Plot 1: Beverloo comparison (log-log) --

fig, ax = plt.subplots(figsize=(7, 5))

# Theory curve (continuous)
D_theory = np.linspace(3 * d, 20 * d, 200)
W_theory = beverloo_2d(D_theory, d, y_depth)
eff_D_theory = D_theory - k_beverloo * d

# Simulation data
eff_D_sim = D_vals - k_beverloo * d

ax.plot(eff_D_theory * 1000, W_theory, "b-", linewidth=2,
        label=f"Beverloo (C={C_beverloo}, k={k_beverloo})")
if len(eff_D_sim) > 0:
    ax.plot(eff_D_sim * 1000, W_meas, "ro", markersize=10, markeredgecolor="k",
            markeredgewidth=1, label="MDDEM simulation")

    # Annotate each point with D/d
    for i, mult in enumerate(mult_labels):
        ax.annotate(f"D={mult}d", (eff_D_sim[i] * 1000, W_meas[i]),
                     textcoords="offset points", xytext=(8, -5), fontsize=9)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Effective orifice width $(D - k \cdot d)$ [mm]", fontsize=13)
ax.set_ylabel(r"Mass flow rate $W$ [kg/s]", fontsize=13)
ax.set_title("Beverloo Hopper Discharge: MDDEM vs Theory (quasi-2D)", fontsize=14)
ax.legend(fontsize=12, loc="upper left")
ax.grid(True, which="both", alpha=0.3)
ax.tick_params(labelsize=11)

# Slope reference
ax.text(0.95, 0.05, "Expected slope: 3/2", transform=ax.transAxes,
        fontsize=10, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

fig.tight_layout()
fig.savefig(os.path.join(script_dir, "beverloo_comparison.png"), dpi=150)
print(f"Saved: {os.path.join(script_dir, 'beverloo_comparison.png')}")
plt.close()

# -- Plot 2: Mass vs time for each orifice --

fig, ax = plt.subplots(figsize=(7, 5))

colors = plt.cm.viridis(np.linspace(0.2, 0.8, max(len(time_data), 1)))
for i, mult in enumerate(sorted(time_data.keys())):
    t, m = time_data[mult]
    if t is not None and len(t) > 0:
        # Offset time so discharge starts at t=0
        t_offset = t - t[0]
        ax.plot(t_offset, m, color=colors[i], linewidth=1.5,
                label=f"D = {mult}d = {mult*d*1000:.0f} mm")

ax.set_xlabel("Time since discharge start [s]", fontsize=13)
ax.set_ylabel("Mass remaining in hopper [kg]", fontsize=13)
ax.set_title("Hopper Discharge: Mass vs Time", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(os.path.join(script_dir, "mass_vs_time.png"), dpi=150)
print(f"Saved: {os.path.join(script_dir, 'mass_vs_time.png')}")
plt.close()

print("Plots generated successfully.")

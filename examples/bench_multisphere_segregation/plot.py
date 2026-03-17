#!/usr/bin/env python3
"""
Plot multisphere segregation benchmark results.

Generates publication-quality figures:
  1. Segregation index S vs time
  2. Center-of-mass height trajectories for spheres vs dimers

Reads: data/segregation.csv
Saves: segregation_index.png, com_trajectories.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, "data", "segregation.csv")

if not os.path.isfile(data_file):
    print(f"ERROR: {data_file} not found. Run simulation first.")
    sys.exit(1)

data = np.loadtxt(data_file, delimiter=",", skiprows=1)
if data.ndim == 1:
    data = data.reshape(1, -1)

steps = data[:, 0]
times = data[:, 1] * 1000  # Convert to ms
z_sphere = data[:, 2] * 1000  # Convert to mm
z_dimer = data[:, 3] * 1000  # Convert to mm
seg_index = data[:, 4]

# ── Figure 1: Segregation Index vs Time ──────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(times, seg_index, "b-", linewidth=1.5, label="Segregation index $S$")
ax.axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)

# Add running average
if len(seg_index) > 20:
    window = max(len(seg_index) // 20, 5)
    running_avg = np.convolve(seg_index, np.ones(window) / window, mode="valid")
    t_avg = times[window - 1:]
    ax.plot(t_avg, running_avg, "r-", linewidth=2.5, alpha=0.8, label=f"Running average (window={window})")

ax.set_xlabel("Time [ms]", fontsize=13)
ax.set_ylabel("Segregation index $S = (z_d - z_s) / (z_d + z_s)$", fontsize=13)
ax.set_title("Sphere/Dimer Segregation Under Vertical Vibration", fontsize=14)
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(os.path.join(script_dir, "segregation_index.png"), dpi=150)
print("Saved segregation_index.png")

# ── Figure 2: COM Trajectories ──────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(times, z_sphere, "bo-", markersize=3, linewidth=1, alpha=0.7, label="Spheres (COM $z$)")
ax2.plot(times, z_dimer, "rs-", markersize=3, linewidth=1, alpha=0.7, label="Dimers (COM $z$)")

ax2.set_xlabel("Time [ms]", fontsize=13)
ax2.set_ylabel("Center-of-mass height [mm]", fontsize=13)
ax2.set_title("COM Height: Spheres vs Dimers", fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=11)

fig2.tight_layout()
fig2.savefig(os.path.join(script_dir, "com_trajectories.png"), dpi=150)
print("Saved com_trajectories.png")

plt.close("all")

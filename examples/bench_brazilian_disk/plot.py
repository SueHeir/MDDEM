#!/usr/bin/env python3
"""
Plot Brazilian disk tensile test results.

Generates two figures:
  1. Load-displacement curve with peak load and tensile strength annotation.
  2. Bond breakage progression overlaid on the load curve.

Usage:
    python3 examples/bench_brazilian_disk/plot.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, "data", "load_displacement.csv")

if not os.path.isfile(data_file):
    print(f"ERROR: {data_file} not found. Run simulation first.")
    sys.exit(1)

data = np.genfromtxt(data_file, delimiter=",", skip_header=1, dtype=float)
if data.ndim == 1:
    data = data.reshape(1, -1)

# Columns: step, time, load, gap, z_bottom, z_top, f_bottom, f_top, bonds_broken, bond_count
time = data[:, 1]
load = data[:, 2]
gap = data[:, 3]
bonds_broken = data[:, 8]
bond_count = data[:, 9]

# Compute displacement from initial gap
initial_gap = gap[0]
displacement = initial_gap - gap  # positive = compression

# Brazilian test parameters
D = 0.02   # disk diameter [m]
t = 0.001  # thickness (one particle diameter in y) [m]

# Peak load
peak_idx = np.argmax(load)
peak_load = load[peak_idx]
sigma_t = 2 * peak_load / (np.pi * D * t) if peak_load > 0 else 0

# ── Figure 1: Load vs Displacement ──────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(displacement * 1e3, load, "b-", linewidth=1.5, label="Simulation")
ax1.plot(
    displacement[peak_idx] * 1e3,
    peak_load,
    "ro",
    markersize=8,
    label=f"Peak = {peak_load:.2e} N",
)

ax1.set_xlabel("Platen displacement [mm]", fontsize=12)
ax1.set_ylabel("Compressive load P [N]", fontsize=12)
ax1.set_title("Brazilian Disk Test — Load vs Displacement", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Annotate tensile strength
if sigma_t > 0:
    ax1.annotate(
        f"$\\sigma_t$ = {sigma_t:.2e} Pa\n(2P / $\\pi$Dt)",
        xy=(displacement[peak_idx] * 1e3, peak_load),
        xytext=(displacement[peak_idx] * 1e3 + 0.02, peak_load * 0.7),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray"),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
    )

fig1.tight_layout()
fig1_path = os.path.join(script_dir, "load_displacement.png")
fig1.savefig(fig1_path, dpi=150)
print(f"Saved: {fig1_path}")

# ── Figure 2: Load + Bond Breakage ──────────────────────────────────────

fig2, ax2a = plt.subplots(figsize=(8, 5))
color1 = "tab:blue"
ax2a.plot(time * 1e3, load, color=color1, linewidth=1.5, label="Load")
ax2a.set_xlabel("Time [ms]", fontsize=12)
ax2a.set_ylabel("Compressive load P [N]", fontsize=12, color=color1)
ax2a.tick_params(axis="y", labelcolor=color1)

ax2b = ax2a.twinx()
color2 = "tab:red"
ax2b.plot(time * 1e3, bonds_broken, color=color2, linewidth=1.5, linestyle="--",
          label="Bonds broken")
ax2b.set_ylabel("Cumulative bonds broken", fontsize=12, color=color2)
ax2b.tick_params(axis="y", labelcolor=color2)

ax2a.set_title("Brazilian Disk Test — Load & Bond Breakage vs Time", fontsize=14)

# Combined legend
lines1, labels1 = ax2a.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2a.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)
ax2a.grid(True, alpha=0.3)

fig2.tight_layout()
fig2_path = os.path.join(script_dir, "load_bond_breakage.png")
fig2.savefig(fig2_path, dpi=150)
print(f"Saved: {fig2_path}")

plt.close("all")
print(f"\nPeak load:        {peak_load:.4e} N")
print(f"Tensile strength: {sigma_t:.4e} Pa")
print(f"Bonds broken:     {int(bonds_broken[-1])}")

#!/usr/bin/env python3
"""
Generate publication-quality plots for the packed bed thermal conductivity benchmark.

Produces three figures:
  1. Temperature profile at steady state (T vs z) with linear fit
  2. Wall heat flux evolution over time (convergence to steady state)
  3. Average temperature evolution over time

Usage:
    cd examples/bench_thermal_bed
    python3 plot.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── Load data ───────────────────────────────────────────────────────────────

profile_file = os.path.join(script_dir, "data", "ThermalProfile.csv")
flux_file = os.path.join(script_dir, "data", "WallHeatFlux.txt")

profile = np.loadtxt(profile_file, delimiter=",", skiprows=1)
z = profile[:, 0] * 1000  # Convert to mm
T = profile[:, 1]

flux_data = np.loadtxt(flux_file, comments="#")
if flux_data.ndim == 1:
    flux_data = flux_data.reshape(1, -1)
flux_time = flux_data[:, 1]
flux_Q = flux_data[:, 2]
flux_Tavg = flux_data[:, 3]

T_hot = 400.0
T_cold = 300.0

# ── Figure 1: Temperature profile at steady state ──────────────────────────

fig1, ax1 = plt.subplots(figsize=(6, 5))

idx = np.argsort(z)
z_s = z[idx]
T_s = T[idx]

coeffs = np.polyfit(z_s, T_s, 1)
z_fit = np.linspace(z_s.min(), z_s.max(), 100)
T_fit = np.polyval(coeffs, z_fit)

T_pred = np.polyval(coeffs, z_s)
ss_res = np.sum((T_s - T_pred) ** 2)
ss_tot = np.sum((T_s - np.mean(T_s)) ** 2)
R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

ax1.scatter(z_s, T_s, s=15, alpha=0.6, color="steelblue",
            edgecolors="navy", linewidths=0.5, label="DEM particles", zorder=3)
ax1.plot(z_fit, T_fit, "r--", lw=2, label=f"Linear fit (R²={R2:.3f})", zorder=4)
ax1.axhline(T_hot, color="orangered", ls=":", alpha=0.5, label=f"T_hot = {T_hot} K")
ax1.axhline(T_cold, color="dodgerblue", ls=":", alpha=0.5, label=f"T_cold = {T_cold} K")

ax1.set_xlabel("Height z [mm]", fontsize=12)
ax1.set_ylabel("Temperature [K]", fontsize=12)
ax1.set_title("Steady-State Temperature Profile", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9, loc="best")
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(script_dir, "temperature_profile.png"), dpi=150)
print("Saved temperature_profile.png")

# ── Figure 2: Wall heat flux evolution ──────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(6, 5))

ax2.plot(flux_time * 1000, flux_Q, "b-", lw=1.2, label="Wall heat flux Q")
n = len(flux_Q)
if n >= 10:
    ss_start = int(0.8 * n)
    Q_ss = np.mean(flux_Q[ss_start:])
    ax2.axhline(Q_ss, color="red", ls="--", lw=1.5, label=f"SS mean = {Q_ss:.4e} W")
    ax2.axvline(flux_time[ss_start] * 1000, color="gray", ls=":", alpha=0.5)

ax2.set_xlabel("Time [ms]", fontsize=12)
ax2.set_ylabel("Wall Heat Flux [W]", fontsize=12)
ax2.set_title("Bottom Wall Heat Flux vs Time", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9, loc="best")
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(script_dir, "wall_heat_flux.png"), dpi=150)
print("Saved wall_heat_flux.png")

# ── Figure 3: Average temperature evolution ─────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(6, 5))

ax3.plot(flux_time * 1000, flux_Tavg, "g-", lw=1.2, label="Avg temperature")
ax3.axhline((T_hot + T_cold) / 2, color="gray", ls="--", alpha=0.7,
            label=f"Midpoint = {(T_hot+T_cold)/2:.0f} K")

ax3.set_xlabel("Time [ms]", fontsize=12)
ax3.set_ylabel("Average Temperature [K]", fontsize=12)
ax3.set_title("Average Particle Temperature vs Time", fontsize=13, fontweight="bold")
ax3.legend(fontsize=9, loc="best")
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(os.path.join(script_dir, "avg_temperature.png"), dpi=150)
print("Saved avg_temperature.png")

plt.close("all")

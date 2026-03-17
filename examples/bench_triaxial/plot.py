#!/usr/bin/env python3
"""
Generate publication-quality plots for the triaxial compression benchmark.

Produces:
  1. stress_strain.png   — Deviatoric stress q vs axial strain for all pressures
  2. mohr_circles.png    — Mohr circles at failure with fitted failure envelope
  3. q_p_plot.png        — q–p diagram with Mohr-Coulomb failure line
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
pressures = ["10kPa", "50kPa", "200kPa"]
colors = ["#1f77b4", "#ff7f0e", "#d62728"]

# ── Load data ──────────────────────────────────────────────────────────────

datasets = {}  # label → structured array
peak_data = {}  # label → (sigma_1_peak, sigma_3_at_peak)

for label in pressures:
    csv_path = os.path.join(script_dir, f"data_{label}", "triaxial_stress.csv")
    if not os.path.isfile(csv_path):
        print(f"  Skipping {label}: {csv_path} not found")
        continue
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        continue
    datasets[label] = data
    idx_peak = np.argmax(data["q"])
    peak_data[label] = (data["sigma_1"][idx_peak], data["sigma_3"][idx_peak])

if not datasets:
    print("ERROR: No data files found. Run the simulations first.")
    sys.exit(1)

# ── Plot 1: Stress–strain curves ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
for (label, data), color in zip(datasets.items(), colors):
    strain_pct = data["strain"] * 100  # convert to percent
    q_kpa = data["q"] / 1000           # convert to kPa
    ax.plot(strain_pct, q_kpa, "-", color=color, linewidth=1.5, label=f"σ₃ = {label}")

ax.set_xlabel("Axial Strain ε₁ [%]", fontsize=12)
ax.set_ylabel("Deviatoric Stress q = σ₁ − σ₃ [kPa]", fontsize=12)
ax.set_title("Triaxial Compression — Stress–Strain Curves", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "stress_strain.png"), dpi=150)
plt.close(fig)
print("  Saved stress_strain.png")

# ── Plot 2: Mohr circles at failure ──────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))
theta = np.linspace(0, np.pi, 200)

for (label, (s1, s3)), color in zip(peak_data.items(), colors):
    center = (s1 + s3) / 2 / 1000  # kPa
    radius = (s1 - s3) / 2 / 1000  # kPa
    sigma = center + radius * np.cos(theta)
    tau = radius * np.sin(theta)
    ax.plot(sigma, tau, "-", color=color, linewidth=1.5, label=f"σ₃ = {label}")
    ax.plot(center, 0, "o", color=color, markersize=4)

# Fit Mohr-Coulomb envelope: τ = c + σ tan(φ)
# From peak stresses, the tangent line touches all circles.
# For cohesionless: sin(φ) = (σ₁-σ₃)/(σ₁+σ₃)
phi_vals = []
for s1, s3 in peak_data.values():
    if (s1 + s3) > 0:
        sin_phi = (s1 - s3) / (s1 + s3)
        sin_phi = max(-1.0, min(1.0, sin_phi))
        phi_vals.append(np.arcsin(sin_phi))

if phi_vals:
    phi_avg = np.mean(phi_vals)
    # Failure envelope: τ = σ tan(φ)  (c = 0)
    sigma_max = max(s1 for s1, _ in peak_data.values()) / 1000 * 1.1
    sigma_line = np.linspace(0, sigma_max, 100)
    tau_line = sigma_line * np.tan(phi_avg)
    ax.plot(sigma_line, tau_line, "k--", linewidth=2,
            label=f"MC envelope: φ = {np.degrees(phi_avg):.1f}°")

ax.set_xlabel("Normal Stress σ [kPa]", fontsize=12)
ax.set_ylabel("Shear Stress τ [kPa]", fontsize=12)
ax.set_title("Mohr Circles at Failure with Mohr-Coulomb Envelope", fontsize=13)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "mohr_circles.png"), dpi=150)
plt.close(fig)
print("  Saved mohr_circles.png")

# ── Plot 3: q–p diagram ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

# Plot peak points
p_peaks = []
q_peaks = []
for (label, (s1, s3)), color in zip(peak_data.items(), colors):
    p_val = (s1 + 2 * s3) / 3 / 1000  # kPa
    q_val = (s1 - s3) / 1000           # kPa
    p_peaks.append(p_val)
    q_peaks.append(q_val)
    ax.plot(p_val, q_val, "o", color=color, markersize=8, zorder=5,
            label=f"σ₃ = {label}")

# Plot stress paths
for (label, data), color in zip(datasets.items(), colors):
    p_path = data["p"] / 1000
    q_path = data["q"] / 1000
    ax.plot(p_path, q_path, "-", color=color, linewidth=1, alpha=0.5)

# Fit q = M p (cohesionless) through peak points
p_arr = np.array(p_peaks)
q_arr = np.array(q_peaks)
if len(p_arr) >= 2 and np.sum(p_arr ** 2) > 0:
    # Least squares fit q = M * p
    M = np.sum(p_arr * q_arr) / np.sum(p_arr ** 2)
    # M = 6 sin(φ) / (3 - sin(φ))  →  sin(φ) = 3M / (6 + M)
    sin_phi_qp = 3 * M / (6 + M)
    sin_phi_qp = max(-1.0, min(1.0, sin_phi_qp))
    phi_qp = np.degrees(np.arcsin(sin_phi_qp))
    p_line = np.linspace(0, max(p_peaks) * 1.2, 100)
    q_line = M * p_line
    ax.plot(p_line, q_line, "k--", linewidth=2,
            label=f"q = {M:.2f} p  (φ = {phi_qp:.1f}°)")

ax.set_xlabel("Mean Stress p = (σ₁ + 2σ₃)/3 [kPa]", fontsize=12)
ax.set_ylabel("Deviatoric Stress q = σ₁ − σ₃ [kPa]", fontsize=12)
ax.set_title("q–p Diagram with Mohr-Coulomb Failure Line", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "q_p_plot.png"), dpi=150)
plt.close(fig)
print("  Saved q_p_plot.png")

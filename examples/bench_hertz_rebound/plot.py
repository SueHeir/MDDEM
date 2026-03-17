#!/usr/bin/env python3
"""
Generate publication-quality plots for the Hertz rebound benchmark.

Produces:
  1. COR measured vs COR input (should be 1:1 line)
  2. Contact duration vs impact velocity compared to Hertz theory
  3. Peak overlap vs impact velocity compared to Hertz theory

Usage (from repo root):
    python3 examples/bench_hertz_rebound/plot.py
"""

import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_FILE = os.path.join(SCRIPT_DIR, "data", "sweep_results.csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

# Material properties
YOUNGS_MOD = 70.0e9
POISSON_RATIO = 0.22
RADIUS = 0.005
DENSITY = 2500.0

E_STAR = YOUNGS_MOD / (2.0 * (1.0 - POISSON_RATIO**2))
MASS = (4.0 / 3.0) * math.pi * RADIUS**3 * DENSITY
M_EFF = MASS
R_EFF = RADIUS

# Plot styling
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox_inches": "tight",
})

MARKERS = ["o", "s", "^", "D"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def hertz_contact_duration(v0):
    return 2.87 * (M_EFF**2 / (R_EFF * E_STAR**2 * v0))**0.2


def hertz_max_overlap(v0):
    return (15.0 * M_EFF * v0**2 / (16.0 * R_EFF**0.5 * E_STAR))**0.4


def main():
    if not os.path.isfile(SWEEP_FILE):
        print(f"ERROR: {SWEEP_FILE} not found. Run sweep first.")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    with open(SWEEP_FILE) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Organize data by COR
    cors = sorted(set(float(r["input_cor"]) for r in rows))
    vels = sorted(set(float(r["input_v0"]) for r in rows))

    data = {}
    for r in rows:
        key = (float(r["input_cor"]), float(r["input_v0"]))
        data[key] = {
            "cor_meas": float(r["cor_measured"]),
            "tc": float(r["contact_time"]),
            "delta_max": float(r["max_overlap"]),
            "v_impact": float(r["v_impact"]),
        }

    # ── Plot 1: COR measured vs COR input ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for iv, v0 in enumerate(vels):
        cor_in = []
        cor_out = []
        for c in cors:
            if (c, v0) in data:
                cor_in.append(c)
                cor_out.append(data[(c, v0)]["cor_meas"])
        ax.plot(cor_in, cor_out, MARKERS[iv], color=COLORS[iv], markersize=8,
                label=f"v = {v0} m/s")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal (1:1)")
    ax.set_xlabel("Input COR")
    ax.set_ylabel("Measured COR")
    ax.set_title("Coefficient of Restitution: Input vs Measured")
    ax.legend(loc="upper left")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(PLOT_DIR, "cor_validation.png"))
    plt.close(fig)
    print(f"Saved: {PLOT_DIR}/cor_validation.png")

    # ── Plot 2: Contact duration vs impact velocity ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    # Theory line (use a range of velocities)
    v_theory = np.linspace(0.08, 2.5, 200)
    tc_theory = np.array([hertz_contact_duration(v) for v in v_theory])
    ax.plot(v_theory, tc_theory * 1e6, "k-", linewidth=2, label="Hertz theory (elastic)")

    for ic, cor in enumerate(cors):
        v_list = []
        tc_list = []
        for v0 in vels:
            if (cor, v0) in data:
                v_list.append(data[(cor, v0)]["v_impact"])
                tc_list.append(data[(cor, v0)]["tc"])
        ax.plot(v_list, [t * 1e6 for t in tc_list], MARKERS[ic], color=COLORS[ic],
                markersize=8, label=f"COR = {cor}")

    ax.set_xlabel("Impact velocity [m/s]")
    ax.set_ylabel("Contact duration [µs]")
    ax.set_title("Contact Duration vs Impact Velocity")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    fig.savefig(os.path.join(PLOT_DIR, "contact_duration.png"))
    plt.close(fig)
    print(f"Saved: {PLOT_DIR}/contact_duration.png")

    # ── Plot 3: Peak overlap vs impact velocity ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    delta_theory = np.array([hertz_max_overlap(v) for v in v_theory])
    ax.plot(v_theory, delta_theory * 1e6, "k-", linewidth=2, label="Hertz theory (elastic)")

    for ic, cor in enumerate(cors):
        v_list = []
        d_list = []
        for v0 in vels:
            if (cor, v0) in data:
                v_list.append(data[(cor, v0)]["v_impact"])
                d_list.append(data[(cor, v0)]["delta_max"])
        ax.plot(v_list, [d * 1e6 for d in d_list], MARKERS[ic], color=COLORS[ic],
                markersize=8, label=f"COR = {cor}")

    ax.set_xlabel("Impact velocity [m/s]")
    ax.set_ylabel("Peak overlap [µm]")
    ax.set_title("Peak Overlap vs Impact Velocity")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    fig.savefig(os.path.join(PLOT_DIR, "peak_overlap.png"))
    plt.close(fig)
    print(f"Saved: {PLOT_DIR}/peak_overlap.png")

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()

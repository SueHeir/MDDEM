#!/usr/bin/env python3
"""
Generate publication-quality plots for the lunar regolith benchmark.

Reads results.csv and produces:
  1. angle_vs_surface_energy.png — Angle of repose vs surface energy (Earth & Lunar)
  2. angle_vs_bond_number.png — Angle vs Bond number (universal scaling)

Usage:
    python examples/bench_lunar_regolith/plot.py
"""

import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.csv")
DATA_RESULTS = os.path.join(SCRIPT_DIR, "data", "results.csv")


def load_results(path):
    """Load results CSV into arrays."""
    labels, gravities, gammas, bond_numbers, angles = [], [], [], [], []
    with open(path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            try:
                angle = float(parts[4])
            except ValueError:
                continue
            if np.isnan(angle):
                continue
            labels.append(parts[0])
            gravities.append(float(parts[1]))
            gammas.append(float(parts[2]))
            bond_numbers.append(float(parts[3]))
            angles.append(angle)
    return (np.array(labels), np.array(gravities), np.array(gammas),
            np.array(bond_numbers), np.array(angles))


def castellanos_trend(bo, theta0=27.0):
    """
    Empirical cohesive angle of repose trend (inspired by Castellanos 2005).

    theta = theta0 + k * Bo / (1 + Bo)
    where theta0 is the non-cohesive angle and k sets the max increase.
    Captures: Bo<<1 -> theta0, Bo~1 -> intermediate, Bo>>1 -> plateau.
    """
    k = 50.0  # Maximum additional angle [deg]
    return theta0 + k * bo / (1.0 + bo)


def main():
    # Find results
    if os.path.isfile(RESULTS_FILE):
        results_path = RESULTS_FILE
    elif os.path.isfile(DATA_RESULTS):
        results_path = DATA_RESULTS
    else:
        print(f"ERROR: No results file found at {RESULTS_FILE}")
        print("Run: python run_benchmark.py first.")
        sys.exit(1)

    labels, gravities, gammas, bond_numbers, angles = load_results(results_path)

    if len(labels) == 0:
        print("ERROR: No valid results to plot.")
        sys.exit(1)

    # --- Plot 1: Angle vs Surface Energy ---
    fig, ax = plt.subplots(figsize=(8, 5.5))

    earth_mask = gravities < -5.0
    lunar_mask = gravities > -5.0

    if np.any(earth_mask):
        idx = np.argsort(gammas[earth_mask])
        ax.plot(gammas[earth_mask][idx] * 1000, angles[earth_mask][idx],
                "s-", color="#2196F3", markersize=8, linewidth=2,
                label="Earth (g = 9.81 m/s$^2$)", markerfacecolor="white",
                markeredgewidth=2)

    if np.any(lunar_mask):
        idx = np.argsort(gammas[lunar_mask])
        ax.plot(gammas[lunar_mask][idx] * 1000, angles[lunar_mask][idx],
                "o-", color="#FF5722", markersize=8, linewidth=2,
                label="Moon (g = 1.62 m/s$^2$)", markerfacecolor="white",
                markeredgewidth=2)

    ax.set_xlabel("Surface Energy, $\\gamma$ [mJ/m$^2$]", fontsize=13)
    ax.set_ylabel("Angle of Repose [deg]", fontsize=13)
    ax.set_title("Cohesive Angle of Repose: Earth vs Moon", fontsize=14)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_ylim(0, 90)
    ax.set_xlim(-2, 55)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    # Annotate Bond numbers
    for i in range(len(labels)):
        if bond_numbers[i] > 0.01:
            offset_y = 3 if gravities[i] < -5 else -5
            ax.annotate(f"Bo={bond_numbers[i]:.2f}",
                        (gammas[i] * 1000, angles[i]),
                        textcoords="offset points", xytext=(5, offset_y),
                        fontsize=8, color="gray")

    fig.tight_layout()
    out1 = os.path.join(SCRIPT_DIR, "angle_vs_surface_energy.png")
    fig.savefig(out1, dpi=150)
    print(f"Saved: {out1}")
    plt.close(fig)

    # --- Plot 2: Angle vs Bond Number (universal scaling) ---
    fig, ax = plt.subplots(figsize=(8, 5.5))

    if np.any(earth_mask):
        ax.scatter(bond_numbers[earth_mask], angles[earth_mask],
                   s=100, marker="s", facecolors="white", edgecolors="#2196F3",
                   linewidths=2, zorder=5, label="DEM — Earth")

    if np.any(lunar_mask):
        ax.scatter(bond_numbers[lunar_mask], angles[lunar_mask],
                   s=100, marker="o", facecolors="white", edgecolors="#FF5722",
                   linewidths=2, zorder=5, label="DEM — Moon")

    # Theoretical / empirical curve
    bo_theory = np.linspace(0, 15, 200)
    theta_theory = castellanos_trend(bo_theory)
    ax.plot(bo_theory, theta_theory, "k--", linewidth=1.5, alpha=0.7,
            label="Empirical trend (Castellanos 2005)")

    ax.set_xlabel("Bond Number, Bo = $F_{adhesion}$ / ($mg$)", fontsize=13)
    ax.set_ylabel("Angle of Repose [deg]", fontsize=13)
    ax.set_title("Angle of Repose vs Granular Bond Number", fontsize=14)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_ylim(0, 90)
    ax.set_xlim(-0.5, max(15, np.max(bond_numbers) * 1.2) if len(bond_numbers) > 0 else 15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    # Regime annotations
    ax.axvspan(-0.5, 1.0, alpha=0.05, color="blue")
    ax.axvspan(1.0, 5.0, alpha=0.05, color="orange")
    ax.axvspan(5.0, 20.0, alpha=0.05, color="red")
    ax.text(0.3, 85, "Gravity\ndominated", fontsize=9, ha="center", color="blue", alpha=0.6)
    ax.text(2.5, 85, "Transitional", fontsize=9, ha="center", color="orange", alpha=0.6)
    ax.text(10.0, 85, "Cohesion\ndominated", fontsize=9, ha="center", color="red", alpha=0.6)

    fig.tight_layout()
    out2 = os.path.join(SCRIPT_DIR, "angle_vs_bond_number.png")
    fig.savefig(out2, dpi=150)
    print(f"Saved: {out2}")
    plt.close(fig)

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()

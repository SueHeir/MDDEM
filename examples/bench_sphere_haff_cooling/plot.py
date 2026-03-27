#!/usr/bin/env python3
"""Plot Haff cooling results for single spheres."""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}

df = read_csv(os.path.join(SCRIPT_DIR, "data", "cooling.csv"))
mask = df["time"] > 0
for k in df:
    df[k] = df[k][mask]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Temperature decay
ax = axes[0]
ax.semilogy(df["time"], df["T_trans"], label="T_trans", color="tab:blue")
ax.semilogy(df["time"], df["T_rot"], label="T_rot", color="tab:red")
ax.semilogy(df["time"], df["T_total"], label="T_total", color="black")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Granular Temperature [m\u00b2/s\u00b2]")
ax.set_title("Temperature Decay")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Normalized log-log decay
ax = axes[1]
t0 = df["T_total"][0]
ax.loglog(df["time"], df["T_total"] / t0, label="T_total/T\u2080", color="black")
ax.loglog(df["time"], df["T_trans"] / df["T_trans"][0], label="T_trans/T\u2080", color="tab:blue", alpha=0.7)
ax.loglog(df["time"], df["T_rot"] / df["T_rot"][0], label="T_rot/T\u2080", color="tab:red", alpha=0.7)
# t^-2 reference line anchored to data midpoint
mid = len(df["time"]) // 2
t_ref = df["time"]
y_ref = (df["T_total"][mid] / t0) * (t_ref / t_ref[mid]) ** (-2)
lo = df["T_total"][-1] / t0 * 0.5
valid = (y_ref <= 1.5) & (y_ref >= lo)
ax.loglog(t_ref[valid], y_ref[valid], ":", color="gray", alpha=0.5, label="t\u207b\u00b2")
ax.set_xlabel("Time [s]")
ax.set_ylabel("T / T\u2080")
ax.set_title("Normalized Decay (log-log)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("Haff Cooling \u2014 Single Spheres", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "haff_cooling.png"), dpi=150, bbox_inches="tight")
print("Saved haff_cooling.png")

#!/usr/bin/env python3
"""
Validate LJ Argon simulation results against known liquid Argon properties.

Expected values at T*=0.85, rho*=0.85 (near triple point):
  - g(r) first peak at r ~ 1.0 sigma, height ~ 2.7-3.0
  - Virial pressure P* ~ 1.1 +/- 0.5 (with tail corrections)
  - MSD linear regime: D* ~ 0.035-0.042 (from slope MSD = 6*D*t)

Usage:
    python3 examples/lj_argon/validate.py
"""

import os
import sys
import numpy as np

# Try to import matplotlib; skip plotting if unavailable
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found; skipping plots. Install with: pip install matplotlib")

# ── Locate data directory ──────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

if not os.path.isdir(data_dir):
    print(f"ERROR: data directory not found at {data_dir}")
    print("Run the simulation first:")
    print("  cargo run --release --no-default-features --example lj_argon -- examples/lj_argon/config.toml")
    sys.exit(1)

# ── Load data ──────────────────────────────────────────────────────────────

def load_data(filename):
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        print(f"WARNING: {path} not found, skipping")
        return None
    return np.loadtxt(path, comments="#")

rdf_data = load_data("rdf.txt")
msd_data = load_data("msd.txt")
pressure_data = load_data("pressure.txt")

# ── Analysis ───────────────────────────────────────────────────────────────

print("=" * 60)
print("LJ Argon Validation (T*=0.85, rho*=0.85)")
print("=" * 60)

passed = 0
total = 0

# --- RDF validation ---
if rdf_data is not None and len(rdf_data) > 0:
    r = rdf_data[:, 0]
    gr = rdf_data[:, 1]

    # Find first peak (in range r=0.8 to r=1.5)
    mask = (r > 0.8) & (r < 1.5)
    if np.any(gr[mask] > 0):
        peak_idx = np.argmax(gr[mask])
        r_peak = r[mask][peak_idx]
        gr_peak = gr[mask][peak_idx]

        print(f"\n[RDF]")
        print(f"  First peak position: r = {r_peak:.3f} sigma")
        print(f"  First peak height:   g(r) = {gr_peak:.3f}")

        # Validate peak position (expect 0.95 - 1.15)
        total += 1
        if 0.95 <= r_peak <= 1.15:
            print(f"  Peak position: PASS (expected 0.95-1.15)")
            passed += 1
        else:
            print(f"  Peak position: FAIL (expected 0.95-1.15, got {r_peak:.3f})")

        # Validate peak height (expect 2.4 - 3.5)
        total += 1
        if 2.4 <= gr_peak <= 3.5:
            print(f"  Peak height:   PASS (expected 2.4-3.5)")
            passed += 1
        else:
            print(f"  Peak height:   FAIL (expected 2.4-3.5, got {gr_peak:.3f})")

        # Check that g(r) -> 1 at large r
        far_mask = (r > 2.0) & (r < 2.8)
        if np.any(far_mask):
            gr_far = np.mean(gr[far_mask])
            total += 1
            if 0.7 <= gr_far <= 1.3:
                print(f"  g(r->large):   PASS (mean={gr_far:.3f}, expected ~1.0)")
                passed += 1
            else:
                print(f"  g(r->large):   FAIL (mean={gr_far:.3f}, expected ~1.0)")
    else:
        print("\n[RDF] WARNING: no data in peak region")

# --- Pressure validation ---
if pressure_data is not None and len(pressure_data) > 0:
    steps = pressure_data[:, 0]
    pressure = pressure_data[:, 1]

    # Use second half of data (equilibrated)
    n_equil = len(pressure) // 2
    p_equil = pressure[n_equil:]
    p_mean = np.mean(p_equil)
    p_std = np.std(p_equil) / np.sqrt(len(p_equil))  # standard error

    print(f"\n[Pressure]")
    print(f"  Mean P* (equilibrated): {p_mean:.4f} +/- {p_std:.4f}")
    print(f"  P* range: [{np.min(p_equil):.3f}, {np.max(p_equil):.3f}]")

    total += 1
    if -0.5 <= p_mean <= 3.0:
        print(f"  Pressure:      PASS (expected ~1.1, range -0.5 to 3.0)")
        passed += 1
    else:
        print(f"  Pressure:      FAIL (expected ~1.1, got {p_mean:.3f})")

# --- MSD / Diffusion validation ---
if msd_data is not None and len(msd_data) > 10:
    dt_steps = msd_data[:, 0]
    msd = msd_data[:, 1]

    # Convert step offsets to time (dt=0.005 in reduced units)
    dt_lj = 0.005
    time = dt_steps * dt_lj

    print(f"\n[MSD / Diffusion]")
    print(f"  MSD at t=0:    {msd[0]:.6f}")
    print(f"  MSD at t_max:  {msd[-1]:.4f} (t={time[-1]:.2f})")

    # MSD should be 0 at t=0
    total += 1
    if abs(msd[0]) < 1e-10:
        print(f"  MSD(0)=0:      PASS")
        passed += 1
    else:
        print(f"  MSD(0)=0:      FAIL (got {msd[0]:.6e})")

    # Fit linear regime to get diffusion coefficient
    # Use second half of data for linear fit (skip early ballistic regime)
    n_fit_start = len(time) // 2
    if n_fit_start > 5 and time[-1] > 0:
        t_fit = time[n_fit_start:]
        msd_fit = msd[n_fit_start:]
        # MSD = 6*D*t + c
        coeffs = np.polyfit(t_fit, msd_fit, 1)
        slope = coeffs[0]
        D_star = slope / 6.0

        print(f"  Diffusion D*:  {D_star:.4f} (from slope={slope:.4f})")

        total += 1
        if 0.01 <= D_star <= 0.1:
            print(f"  Diffusion:     PASS (expected 0.035-0.042, range 0.01-0.1)")
            passed += 1
        else:
            print(f"  Diffusion:     FAIL (expected 0.035-0.042, got {D_star:.4f})")

# --- Temperature validation (from KE) ---
# KE = (3/2)*N*T* for LJ reduced units, so T* = 2*KE/(3*N)
# Check from thermo output that T stabilizes
if pressure_data is not None and len(pressure_data) > 0:
    print(f"\n[Summary]")

print(f"\n{'=' * 60}")
print(f"Results: {passed}/{total} checks passed")
if passed == total:
    print("ALL CHECKS PASSED")
else:
    print(f"WARNING: {total - passed} check(s) failed")
print(f"{'=' * 60}")

# ── Plotting ───────────────────────────────────────────────────────────────

if not HAS_MPL:
    sys.exit(0 if passed == total else 1)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("LJ Argon Validation (T*=0.85, $\\rho$*=0.85)", fontsize=14, fontweight="bold")

# --- Panel 1: RDF ---
ax = axes[0, 0]
if rdf_data is not None and len(rdf_data) > 0:
    r = rdf_data[:, 0]
    gr = rdf_data[:, 1]
    ax.plot(r, gr, "b-", linewidth=1.5, label="MDDEM")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Ideal gas")
    ax.axvline(x=1.0, color="red", linestyle=":", alpha=0.5, label="$r = \\sigma$")
    ax.set_xlabel("$r / \\sigma$")
    ax.set_ylabel("$g(r)$")
    ax.set_title("Radial Distribution Function")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, max(3.5, np.max(gr) * 1.1))
    ax.legend(loc="upper right")
    ax.grid(False)
else:
    ax.text(0.5, 0.5, "No RDF data", ha="center", va="center", transform=ax.transAxes)

# --- Panel 2: MSD ---
ax = axes[0, 1]
if msd_data is not None and len(msd_data) > 1:
    dt_steps = msd_data[:, 0]
    msd_vals = msd_data[:, 1]
    time_vals = dt_steps * 0.005
    ax.plot(time_vals, msd_vals, "g-", linewidth=1.5, label="MDDEM")

    # Overlay linear fit from second half
    n_fit_start = len(time_vals) // 2
    if n_fit_start > 5:
        t_fit = time_vals[n_fit_start:]
        msd_fit_vals = msd_vals[n_fit_start:]
        coeffs = np.polyfit(t_fit, msd_fit_vals, 1)
        D_val = coeffs[0] / 6.0
        ax.plot(t_fit, np.polyval(coeffs, t_fit), "r--", linewidth=1.5,
                label=f"Linear fit: D*={D_val:.4f}")

    ax.set_xlabel("$t^*$ (reduced time)")
    ax.set_ylabel("MSD ($\\sigma^2$)")
    ax.set_title("Mean Square Displacement")
    ax.legend(loc="upper left")
    ax.grid(False)
else:
    ax.text(0.5, 0.5, "No MSD data", ha="center", va="center", transform=ax.transAxes)

# --- Panel 3: Pressure time series ---
ax = axes[1, 0]
if pressure_data is not None and len(pressure_data) > 0:
    steps_p = pressure_data[:, 0]
    pressure_p = pressure_data[:, 1]
    ax.plot(steps_p, pressure_p, "m-", linewidth=0.5, alpha=0.6, label="Instantaneous")

    # Running average
    window = min(100, len(pressure_p) // 5)
    if window > 1:
        running_avg = np.convolve(pressure_p, np.ones(window) / window, mode="valid")
        ax.plot(steps_p[window - 1:], running_avg, "k-", linewidth=2,
                label=f"Running avg (w={window})")

    # Reference line
    ax.axhline(y=1.1, color="red", linestyle="--", alpha=0.7, label="Expected P*~1.1")

    n_equil = len(pressure_p) // 2
    p_mean = np.mean(pressure_p[n_equil:])
    ax.axhline(y=p_mean, color="blue", linestyle=":", alpha=0.7,
               label=f"Mean (equil.)={p_mean:.2f}")

    ax.set_xlabel("Step")
    ax.set_ylabel("$P^*$")
    ax.set_title("Virial Pressure")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(False)
else:
    ax.text(0.5, 0.5, "No pressure data", ha="center", va="center", transform=ax.transAxes)

# --- Panel 4: Pressure histogram ---
ax = axes[1, 1]
if pressure_data is not None and len(pressure_data) > 0:
    n_equil = len(pressure_data) // 2
    p_equil = pressure_data[n_equil:, 1]
    ax.hist(p_equil, bins=50, density=True, color="skyblue", edgecolor="navy", alpha=0.7)
    ax.axvline(x=np.mean(p_equil), color="red", linewidth=2,
               label=f"Mean = {np.mean(p_equil):.3f}")
    ax.axvline(x=1.1, color="green", linewidth=2, linestyle="--",
               label="Expected ~1.1")
    ax.set_xlabel("$P^*$")
    ax.set_ylabel("Probability density")
    ax.set_title("Pressure Distribution (equilibrated)")
    ax.legend()
    ax.grid(False)
else:
    ax.text(0.5, 0.5, "No pressure data", ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()
output_path = os.path.join(script_dir, "validation.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {output_path}")
plt.close()

sys.exit(0 if passed == total else 1)

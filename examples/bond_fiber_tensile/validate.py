#!/usr/bin/env python3
"""
Validate BPM fiber tensile result: fit σ = E · ε and compare to input E.

Reads data/fiber_tensile.csv produced by the Rust example and prints the
fitted slope. Optionally plots σ(ε) if matplotlib is available.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path


def load(path: str):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def linear_fit_through_origin(x, y):
    """Least-squares slope for y = m·x (no intercept)."""
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    if sxx == 0.0:
        return 0.0
    return sxy / sxx


def main():
    default = Path(__file__).parent / "data" / "fiber_tensile.csv"
    path = sys.argv[1] if len(sys.argv) > 1 else str(default)
    if not os.path.exists(path):
        sys.exit(f"CSV not found: {path}. Run the Rust example first.")

    rows = load(path)
    if not rows:
        sys.exit("CSV has no data rows")

    eps = [r["strain_global"] for r in rows]
    sig = [r["stress_mid"] for r in rows]
    eps_local = [r["strain_mid"] for r in rows]

    # Drop initial transient (first 10% of samples) and rows with ε ≈ 0
    ntrim = max(1, len(rows) // 10)
    eps_f = eps[ntrim:]
    sig_f = sig[ntrim:]

    e_fit = linear_fit_through_origin(eps_f, sig_f)

    # ε_local vs ε_global should agree at steady state (uniform strain distribution).
    ratios = [el / eg for el, eg in zip(eps_local[ntrim:], eps_f) if abs(eg) > 1e-8]
    mean_ratio = sum(ratios) / len(ratios) if ratios else float("nan")

    k_n = rows[-1]["k_n"]
    area = rows[-1]["area"]
    length0 = rows[-1]["length0"]
    bond_len = length0 / 10.0
    e_from_k = k_n * bond_len / area  # invert K_n = E·A/L

    print("=== BPM Fiber Tensile Validation ===")
    print(f"  samples         : {len(rows)} rows (using last {len(eps_f)})")
    print(f"  global ε range  : {min(eps):.4e} → {max(eps):.4e}")
    print(f"  bond_length L   : {bond_len:.6e} m")
    print(f"  bond area A     : {area:.6e} m²")
    print(f"  E from K_n·L/A  : {e_from_k:.6e} Pa  (config input)")
    print(f"  E fit σ/ε       : {e_fit:.6e} Pa")
    print(f"  relative error  : {abs(e_fit - e_from_k) / e_from_k:.3%}")
    print(f"  ε_local / ε_glob: {mean_ratio:.6f}   (1.0 = uniform strain)")

    # Optional plot
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([e * 100 for e in eps], [s / 1e6 for s in sig], "o", ms=3, label="measured")
        eps_fit = sorted(eps_f)
        sig_fit = [e_fit * e for e in eps_fit]
        ax.plot(
            [e * 100 for e in eps_fit],
            [s / 1e6 for s in sig_fit],
            "-",
            label=f"fit E = {e_fit:.3e} Pa",
        )
        ax.set_xlabel("strain ε  (%)")
        ax.set_ylabel("stress σ  (MPa)")
        ax.set_title("BPM fiber: σ(ε)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out = Path(path).parent / "fiber_stress_strain.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  plot saved      : {out}")
    except ImportError:
        print("  (matplotlib not available — skipping plot)")


if __name__ == "__main__":
    main()

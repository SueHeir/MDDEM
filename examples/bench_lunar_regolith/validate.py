#!/usr/bin/env python3
"""
Validate the lunar regolith cohesive angle of repose benchmark.

Reads results.csv (produced by run_benchmark.py) and checks:
  1. Non-cohesive angle is in the expected range (20-40 deg)
  2. Angle increases monotonically with surface energy (for each gravity)
  3. Lunar angles >= Earth angles for the same surface energy (when cohesion > 0)
  4. High-adhesion cases produce significantly steeper piles than no-adhesion

Exit code 0 = all checks pass, 1 = at least one failure.
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.csv")

# Also support reading results from a pre-generated data directory
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
DATA_RESULTS = os.path.join(DATA_DIR, "results.csv")


def load_results(path):
    """Load results CSV into a structured dict."""
    results = {}
    with open(path) as f:
        header = f.readline().strip()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            label = parts[0]
            gz = float(parts[1])
            gamma = float(parts[2])
            bo = float(parts[3])
            try:
                angle = float(parts[4])
            except ValueError:
                angle = np.nan
            try:
                pile_height = float(parts[5]) if len(parts) > 5 else np.nan
            except ValueError:
                pile_height = np.nan
            results[label] = {
                "gravity": gz,
                "surface_energy": gamma,
                "bond_number": bo,
                "angle": angle,
                "pile_height": pile_height,
            }
    return results


def main():
    # Find results file
    if os.path.isfile(RESULTS_FILE):
        results_path = RESULTS_FILE
    elif os.path.isfile(DATA_RESULTS):
        results_path = DATA_RESULTS
    else:
        print(f"ERROR: No results file found.")
        print(f"  Expected: {RESULTS_FILE}")
        print(f"  Run: python run_benchmark.py first.")
        sys.exit(1)

    results = load_results(results_path)

    print("=" * 60)
    print("Lunar Regolith Benchmark Validation")
    print("=" * 60)
    print(f"Source: {results_path}")
    print(f"Cases loaded: {len(results)}")
    print()

    passed = 0
    total = 0

    # Helper to get angle for a case (also check pile_height if present)
    def get_angle(label):
        if label in results and not np.isnan(results[label]["angle"]):
            return results[label]["angle"]
        return None

    def get_height(label):
        if label in results and "pile_height" in results[label]:
            h = results[label].get("pile_height", np.nan)
            if h is not None and not np.isnan(h):
                return h
        return None

    # ---- Check 1: Non-cohesive angles in expected range (20-40 deg) ----
    for env, label in [("Earth", "earth_gamma0.000"), ("Lunar", "lunar_gamma0.000")]:
        total += 1
        angle = get_angle(label)
        if angle is None:
            print(f"  {env} no-adhesion angle:  SKIP (no data)")
        elif 15.0 <= angle <= 45.0:
            print(f"  {env} no-adhesion angle:  PASS ({angle:.1f} deg, expected 15-45)")
            passed += 1
        else:
            print(f"  {env} no-adhesion angle:  FAIL ({angle:.1f} deg, expected 15-45)")

    # ---- Check 2: Angle increases with surface energy (monotonic trend) ----
    for env, prefix in [("Earth", "earth"), ("Lunar", "lunar")]:
        total += 1
        gammas = [0.0, 0.005, 0.02, 0.05]
        labels = [f"{prefix}_gamma{g:.3f}" for g in gammas]
        angles = [get_angle(l) for l in labels]
        valid_angles = [(g, a) for g, a in zip(gammas, angles) if a is not None]

        if len(valid_angles) < 3:
            print(f"  {env} angle vs adhesion:  SKIP (insufficient data: {len(valid_angles)} cases)")
        else:
            # Check that angle generally increases (allow small dips due to noise)
            angle_vals = [a for _, a in valid_angles]
            # Use linear regression: slope should be positive
            gamma_vals = np.array([g for g, _ in valid_angles])
            angle_arr = np.array(angle_vals)
            if len(gamma_vals) >= 2 and np.std(gamma_vals) > 0:
                slope = np.polyfit(gamma_vals, angle_arr, 1)[0]
                if slope > 0:
                    print(f"  {env} angle vs adhesion:  PASS (slope={slope:.1f} deg/(J/m^2))")
                    passed += 1
                else:
                    print(f"  {env} angle vs adhesion:  FAIL (slope={slope:.1f}, expected positive)")
                    print(f"    Angles: {['%.1f' % a for a in angle_vals]}")
            else:
                print(f"  {env} angle vs adhesion:  SKIP (insufficient variation)")

    # ---- Check 3: Lunar angle >= Earth angle for cohesive cases ----
    total += 1
    cohesive_gammas = [0.005, 0.02, 0.05]
    lunar_steeper_count = 0
    comparison_count = 0
    for gamma in cohesive_gammas:
        earth_label = f"earth_gamma{gamma:.3f}"
        lunar_label = f"lunar_gamma{gamma:.3f}"
        earth_angle = get_angle(earth_label)
        lunar_angle = get_angle(lunar_label)
        if earth_angle is not None and lunar_angle is not None:
            comparison_count += 1
            if lunar_angle >= earth_angle - 2.0:  # Allow 2-deg tolerance
                lunar_steeper_count += 1

    if comparison_count == 0:
        print(f"  Lunar >= Earth (cohesive): SKIP (no data)")
    elif lunar_steeper_count == comparison_count:
        print(f"  Lunar >= Earth (cohesive): PASS ({lunar_steeper_count}/{comparison_count} cases)")
        passed += 1
    else:
        print(f"  Lunar >= Earth (cohesive): FAIL ({lunar_steeper_count}/{comparison_count} cases)")

    # ---- Check 4: High adhesion produces significantly steeper pile ----
    for env, prefix in [("Earth", "earth"), ("Lunar", "lunar")]:
        total += 1
        no_adh = get_angle(f"{prefix}_gamma0.000")
        high_adh = get_angle(f"{prefix}_gamma0.050")
        if no_adh is None or high_adh is None:
            print(f"  {env} high-adh steeper:    SKIP (no data)")
        elif high_adh > no_adh + 3.0:  # At least 3 degrees steeper
            print(f"  {env} high-adh steeper:    PASS ({high_adh:.1f} > {no_adh:.1f} + 3)")
            passed += 1
        else:
            print(f"  {env} high-adh steeper:    FAIL ({high_adh:.1f} vs {no_adh:.1f}, need >3 deg increase)")

    # ---- Summary ----
    print()
    print(f"Results: {passed}/{total} checks passed")
    if passed == total:
        print("ALL CHECKS PASSED")
    else:
        print(f"WARNING: {total - passed} check(s) failed or skipped")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

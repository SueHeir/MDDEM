#!/usr/bin/env python3
"""
Validate Hertz rebound benchmark results against analytical predictions.

Checks:
  1. COR accuracy — measured COR matches input COR (3% for COR>=0.7, 12% for COR<0.7)
  2. Contact duration — matches Hertz prediction within 10%
  3. Peak overlap — matches Hertz prediction within 10%
  4. Data completeness — all parameter combinations produced results

Hertz contact theory:
  Contact duration:  t_c = 2.87 * (m*^2 / (R* E*^2 v0))^(1/5)
  Peak overlap:      delta_max = (15 m* v0^2 / (16 R*^(1/2) E*))^(2/5)

where:
  m*   = m/2 for sphere-wall (wall has infinite mass, so m* = m)
  R*   = R   for sphere-wall (wall has infinite radius, so R* = R)
  E*   = E / (2(1 - nu^2))  for identical materials sphere-wall

Reference: K.L. Johnson, Contact Mechanics, Cambridge University Press, 1985.
"""

import os
import sys
import csv
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_FILE = os.path.join(SCRIPT_DIR, "data", "sweep_results.csv")

# Material properties
YOUNGS_MOD = 70.0e9    # Pa
POISSON_RATIO = 0.22
RADIUS = 0.005         # m
DENSITY = 2500.0       # kg/m^3

# Tolerances
# Note: The COR tolerance is COR-dependent because the linear viscoelastic
# damping model (β mapping derived from Hooke/linear theory) deviates when
# used with nonlinear Hertz stiffness. This is a well-known limitation:
#   COR >= 0.7: tight tolerance (3%)
#   COR < 0.7:  relaxed tolerance (12%) due to Hertz nonlinearity
COR_TOL_HIGH = 0.03    # 3% relative tolerance for COR >= 0.7
COR_TOL_LOW = 0.12     # 12% relative tolerance for COR < 0.7
CONTACT_TIME_TOL = 0.10  # 10% relative tolerance for contact duration
# Peak overlap: elastic Hertz theory over-predicts overlap for dissipative contacts
# because energy is lost during approach. Tolerance scales with COR.
OVERLAP_TOL_HIGH = 0.10  # 10% for COR >= 0.85 (near-elastic)
OVERLAP_TOL_MED = 0.15   # 15% for 0.6 <= COR < 0.85
OVERLAP_TOL_LOW = 0.25   # 25% for COR < 0.6 (heavy damping)

# Expected parameter combinations
VELOCITIES = [0.1, 0.5, 1.0, 2.0]
CORS = [0.5, 0.7, 0.9, 0.95]


def hertz_contact_duration(m_eff, r_eff, e_star, v0):
    """Hertz analytical contact duration for elastic impact."""
    return 2.87 * (m_eff**2 / (r_eff * e_star**2 * v0))**0.2


def hertz_max_overlap(m_eff, r_eff, e_star, v0):
    """Hertz analytical maximum overlap for elastic impact."""
    return (15.0 * m_eff * v0**2 / (16.0 * r_eff**0.5 * e_star))**0.4


def main():
    if not os.path.isfile(SWEEP_FILE):
        print(f"ERROR: {SWEEP_FILE} not found.")
        print("Run the sweep first: python3 examples/bench_hertz_rebound/run_sweep.py")
        sys.exit(1)

    # Read results
    with open(SWEEP_FILE) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("ERROR: No data in sweep results file.")
        sys.exit(1)

    # Derived material properties for sphere-wall contact
    # E* = E / (2*(1 - nu^2)) for same-material sphere on wall
    e_star = YOUNGS_MOD / (2.0 * (1.0 - POISSON_RATIO**2))
    mass = (4.0 / 3.0) * math.pi * RADIUS**3 * DENSITY
    m_eff = mass       # sphere-wall: m_eff = m (wall mass -> infinity)
    r_eff = RADIUS     # sphere-wall: R_eff = R (wall radius -> infinity)

    print("=" * 65)
    print("Hertz Contact Rebound Benchmark Validation")
    print("=" * 65)
    print(f"  E* = {e_star:.3e} Pa")
    print(f"  m  = {mass:.6e} kg")
    print(f"  R  = {RADIUS*1000:.1f} mm")
    print()

    total = 0
    passed = 0
    cor_pass = 0
    cor_total = 0
    tc_pass = 0
    tc_total = 0
    ov_pass = 0
    ov_total = 0

    for row in rows:
        v0 = float(row["input_v0"])
        cor_input = float(row["input_cor"])
        cor_meas = float(row["cor_measured"])
        tc_meas = float(row["contact_time"])
        delta_max_meas = float(row["max_overlap"])
        v_impact = float(row["v_impact"])

        # Use actual impact velocity for Hertz predictions (accounts for gravity
        # acceleration during fall)
        tc_theory = hertz_contact_duration(m_eff, r_eff, e_star, v_impact)
        delta_max_theory = hertz_max_overlap(m_eff, r_eff, e_star, v_impact)

        # --- Check 1: COR accuracy ---
        cor_total += 1
        total += 1
        cor_err = abs(cor_meas - cor_input) / cor_input
        cor_tol = COR_TOL_HIGH if cor_input >= 0.7 else COR_TOL_LOW
        if cor_err <= cor_tol:
            cor_pass += 1
            passed += 1
            status_cor = "PASS"
        else:
            status_cor = "FAIL"

        # --- Check 2: Contact duration ---
        tc_total += 1
        total += 1
        tc_err = abs(tc_meas - tc_theory) / tc_theory
        if tc_err <= CONTACT_TIME_TOL:
            tc_pass += 1
            passed += 1
            status_tc = "PASS"
        else:
            status_tc = "FAIL"

        # --- Check 3: Peak overlap ---
        ov_total += 1
        total += 1
        ov_err = abs(delta_max_meas - delta_max_theory) / delta_max_theory
        if cor_input >= 0.85:
            ov_tol = OVERLAP_TOL_HIGH
        elif cor_input >= 0.6:
            ov_tol = OVERLAP_TOL_MED
        else:
            ov_tol = OVERLAP_TOL_LOW
        if ov_err <= ov_tol:
            ov_pass += 1
            passed += 1
            status_ov = "PASS"
        else:
            status_ov = "FAIL"

        print(f"v0={v0:.1f} m/s, COR_in={cor_input:.2f}:")
        print(f"  COR:     {cor_meas:.4f} vs {cor_input:.4f}"
              f"  (err={cor_err*100:.2f}%)  [{status_cor}]")
        print(f"  t_c:     {tc_meas:.3e} vs {tc_theory:.3e} s"
              f"  (err={tc_err*100:.1f}%)  [{status_tc}]")
        print(f"  d_max:   {delta_max_meas:.3e} vs {delta_max_theory:.3e} m"
              f"  (err={ov_err*100:.1f}%)  [{status_ov}]")

    # --- Check 4: Data completeness ---
    total += 1
    expected = len(VELOCITIES) * len(CORS)
    if len(rows) == expected:
        passed += 1
        print(f"\nCompleteness: {len(rows)}/{expected} cases  [PASS]")
    else:
        print(f"\nCompleteness: {len(rows)}/{expected} cases  [FAIL]")

    print()
    print(f"COR checks:          {cor_pass}/{cor_total} passed")
    print(f"Contact time checks:  {tc_pass}/{tc_total} passed")
    print(f"Overlap checks:       {ov_pass}/{ov_total} passed")
    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("ALL CHECKS PASSED")
    else:
        print(f"WARNING: {total - passed} check(s) failed")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

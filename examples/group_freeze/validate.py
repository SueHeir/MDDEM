#!/usr/bin/env python3
"""Validate group_freeze example output.

Checks:
1. Frozen atoms (z < 3.0) have ~zero velocity in dump files.
2. Global temperature < T* (frozen atoms drag average down).
"""

import csv
import glob
import sys
import os

def main():
    # Find the latest dump file (look relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dump_dir = os.path.join(script_dir, "dump")
    if not os.path.isdir(dump_dir):
        print(f"FAIL: dump directory '{dump_dir}' not found")
        sys.exit(1)

    dump_files = sorted(glob.glob(f"{dump_dir}/dump_*_rank0.csv"))
    if not dump_files:
        print("FAIL: no dump files found")
        sys.exit(1)

    # Check the last dump file
    last_dump = dump_files[-1]
    print(f"Checking {last_dump}")

    frozen_max_v = 0.0
    mobile_ke_sum = 0.0
    mobile_count = 0

    with open(last_dump) as f:
        reader = csv.DictReader(f)
        for row in reader:
            z = float(row["z"])
            vx = float(row["vx"])
            vy = float(row["vy"])
            vz = float(row["vz"])
            v2 = vx*vx + vy*vy + vz*vz
            if z < 3.0:
                frozen_max_v = max(frozen_max_v, v2**0.5)
            else:
                mobile_ke_sum += 0.5 * 1.0 * v2  # mass = 1.0
                mobile_count += 1

    ok = True

    # Frozen atoms should have zero velocity
    if frozen_max_v > 1e-10:
        print(f"FAIL: frozen atoms have max |v| = {frozen_max_v:.2e} (expected ~0)")
        ok = False
    else:
        print(f"OK: frozen atoms max |v| = {frozen_max_v:.2e}")

    # Mobile temperature should be near T*=0.85
    if mobile_count > 3:
        ndof = 3.0 * mobile_count - 3.0
        mobile_temp = 2.0 * mobile_ke_sum / ndof
        if abs(mobile_temp - 0.85) > 0.3:
            print(f"FAIL: mobile T* = {mobile_temp:.4f} (expected ~0.85)")
            ok = False
        else:
            print(f"OK: mobile T* = {mobile_temp:.4f}")
    else:
        print("WARN: too few mobile atoms to check temperature")

    if ok:
        print("PASS")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate poiseuille_flow example: check wall atoms are frozen and fluid has velocity."""

import sys
import os
import csv

# Find the latest dump files
dump_dir = "examples/poiseuille_flow/dump"
if not os.path.isdir(dump_dir):
    print(f"ERROR: dump directory not found: {dump_dir}")
    sys.exit(1)

dump_files = sorted(
    [f for f in os.listdir(dump_dir) if f.endswith(".csv")],
    key=lambda f: int(f.split("_")[1]),
)

if not dump_files:
    print("ERROR: No dump files found")
    sys.exit(1)

# Read the last dump file
last_dump = os.path.join(dump_dir, dump_files[-1])
print(f"Reading: {last_dump}")

wall_vx = []
fluid_vx = []

with open(last_dump) as f:
    reader = csv.DictReader(f)
    for row in reader:
        z = float(row["z"])
        vx = float(row["vx"])
        if z < 2.5 or z > 7.5:
            wall_vx.append(abs(vx))
        else:
            fluid_vx.append(vx)

if wall_vx:
    max_wall_v = max(wall_vx)
    print(f"Max wall atom |vx|: {max_wall_v:.6f}")
    if max_wall_v < 1e-10:
        print("PASS: Wall atoms are frozen")
    else:
        print(f"FAIL: Wall atoms have velocity {max_wall_v}")

if fluid_vx:
    avg_fluid_vx = sum(fluid_vx) / len(fluid_vx)
    print(f"Average fluid vx: {avg_fluid_vx:.6f}")
    if avg_fluid_vx > 0:
        print("PASS: Fluid has positive vx (driven by addforce)")
    else:
        print("WARNING: Fluid average vx is not positive (may need more steps)")

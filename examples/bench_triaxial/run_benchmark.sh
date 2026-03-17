#!/usr/bin/env bash
# Run the triaxial compression benchmark at all four confining pressures.
#
# Usage:
#   cd <repo_root>
#   bash examples/bench_triaxial/run_benchmark.sh
#
# After completion, run validation and plotting:
#   python3 examples/bench_triaxial/validate.py
#   python3 examples/bench_triaxial/plot.py

set -euo pipefail

EXAMPLE_DIR="examples/bench_triaxial"

for pressure in 10kPa 50kPa 100kPa 200kPa; do
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Running triaxial compression at σ₃ = ${pressure}"
    echo "══════════════════════════════════════════════════════════════"
    cargo run --release --example bench_triaxial --no-default-features \
        -- "${EXAMPLE_DIR}/config_${pressure}.toml"
done

echo ""
echo "All simulations complete."
echo "Run: python3 ${EXAMPLE_DIR}/validate.py"
echo "Run: python3 ${EXAMPLE_DIR}/plot.py"

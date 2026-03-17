#!/usr/bin/env bash
# Run the Beverloo hopper benchmark for multiple orifice widths.
# Each run uses a separate config file generated from the template.
#
# Usage: bash examples/bench_beverloo_hopper/run_sweep.sh
#
# Orifice widths: 6d, 10d, 15d where d = 0.002 m
# Total runtime target: < 5 minutes (3 runs × ~1 min each)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CONFIG="$SCRIPT_DIR/config.toml"

# Particle diameter [m]
D_PARTICLE=0.002

# Orifice widths in units of d (3 widths for good Beverloo fit)
ORIFICE_MULT=(6 10 15)

# Create data collection directory
mkdir -p "$SCRIPT_DIR/data"

for mult in "${ORIFICE_MULT[@]}"; do
    ORIFICE_WIDTH=$(echo "$mult * $D_PARTICLE" | bc -l)
    HALF_ORIFICE=$(echo "$ORIFICE_WIDTH / 2.0" | bc -l)
    # Trim leading zeros for sed compatibility
    HALF_ORIFICE=$(printf "%.4f" "$HALF_ORIFICE")

    # Create output directory
    OUT_DIR="$SCRIPT_DIR/output_D${mult}d"
    mkdir -p "$OUT_DIR"

    # Generate config with modified orifice width
    CONFIG="$OUT_DIR/config.toml"
    sed \
        -e "s/bound_x_high = -0.010/bound_x_high = -${HALF_ORIFICE}/" \
        -e "s/bound_x_low = 0.010/bound_x_low = ${HALF_ORIFICE}/" \
        -e "s|bound_x_low = -0.010|bound_x_low = -${HALF_ORIFICE}|" \
        -e "s|bound_x_high = 0.010|bound_x_high = ${HALF_ORIFICE}|" \
        -e "s|dir = \"examples/bench_beverloo_hopper\"|dir = \"$OUT_DIR\"|" \
        "$BASE_CONFIG" > "$CONFIG"

    echo "=== Running D = ${mult}d = ${ORIFICE_WIDTH} m (half = ${HALF_ORIFICE} m) ==="
    echo "    Config: $CONFIG"
    echo "    Output: $OUT_DIR"

    cargo run --release --example bench_beverloo_hopper --no-default-features \
        -- "$CONFIG"

    # Copy the particle_count.txt with a descriptive name for analysis
    if [ -f "$OUT_DIR/data/particle_count.txt" ]; then
        cp "$OUT_DIR/data/particle_count.txt" "$SCRIPT_DIR/data/particle_count_D${mult}d.txt"
        echo "    Data copied to data/particle_count_D${mult}d.txt"
    else
        echo "    WARNING: No particle_count.txt found in $OUT_DIR/data/"
    fi

    echo ""
done

echo "All runs complete."
echo "Run validation:  python3 $SCRIPT_DIR/validate.py"
echo "Generate plots:  python3 $SCRIPT_DIR/plot.py"

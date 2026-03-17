#!/usr/bin/env bash
# Run the Beverloo hopper benchmark for multiple orifice widths.
# Each run uses a separate config file generated from the template.
#
# Usage: bash examples/bench_beverloo_hopper/run_sweep.sh
#
# Orifice widths: 5d, 8d, 12d, 16d where d = 0.002 m

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CONFIG="$SCRIPT_DIR/config.toml"

# Particle diameter
D_PARTICLE=0.002

# Orifice widths in units of d
ORIFICE_MULT=(5 8 12 16)

for mult in "${ORIFICE_MULT[@]}"; do
    ORIFICE_WIDTH=$(echo "$mult * $D_PARTICLE" | bc -l)
    HALF_ORIFICE=$(echo "$ORIFICE_WIDTH / 2.0" | bc -l)

    # Create output directory
    OUT_DIR="$SCRIPT_DIR/output_D${mult}d"
    mkdir -p "$OUT_DIR"

    # Generate config with modified orifice width
    CONFIG="$OUT_DIR/config.toml"
    sed \
        -e "s/bound_x_high = -0.008/bound_x_high = -${HALF_ORIFICE}/" \
        -e "s/bound_x_low = 0.008/bound_x_low = ${HALF_ORIFICE}/" \
        -e "s|bound_x_low = -0.008|bound_x_low = -${HALF_ORIFICE}|" \
        -e "s|bound_x_high = 0.008|bound_x_high = ${HALF_ORIFICE}|" \
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
    fi

    echo ""
done

echo "All runs complete. Run validate.py and plot.py for analysis."

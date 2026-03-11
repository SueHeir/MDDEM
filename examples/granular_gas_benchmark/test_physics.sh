#!/bin/bash
# DEM Granular Gas Physics Test Script
# Tests the current state of the DEM physics by running simulations
# and validating against Haff's cooling law.
#
# Usage:
#   ./test_physics.sh              # Quick 100k step test
#   ./test_physics.sh full         # Full 2M step validation + Haff plot
#   ./test_physics.sh check        # Just check existing data (no sim)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

MODE="${1:-quick}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

banner() { echo -e "\n${CYAN}════════════════════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}════════════════════════════════════════════════════${NC}\n"; }
pass()   { echo -e "  ${GREEN}✓ $1${NC}"; }
fail()   { echo -e "  ${RED}✗ $1${NC}"; }
info()   { echo -e "  ${YELLOW}→ $1${NC}"; }

# ─── Build ────────────────────────────────────────────────────────────────────

build_sim() {
    banner "Building (release, single-core)"
    cd "$REPO_ROOT"
    cargo build --release --no-default-features --example granular_gas_benchmark 2>&1 | tail -3
    pass "Build succeeded"
}

# ─── Run simulation ──────────────────────────────────────────────────────────

run_sim() {
    local config="$1"
    local label="$2"

    banner "Running: $label"
    info "Config: $config"
    info "Output: $DATA_DIR/"

    mkdir -p "$DATA_DIR"

    cd "$REPO_ROOT"
    local start_time=$(date +%s)

    # Use relative path so output_dir doesn't double the absolute path
    local rel_config
    rel_config=$(python3 -c "import os; print(os.path.relpath('$config', '$REPO_ROOT'))")

    set +e
    "$REPO_ROOT/target/release/examples/granular_gas_benchmark" "$rel_config" 2>&1 | tee "$SCRIPT_DIR/last_run.log"
    local exit_code=${PIPESTATUS[0]}
    set -e

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        pass "Simulation completed in ${elapsed}s"
        return 0
    else
        fail "Simulation CRASHED (exit code $exit_code) after ${elapsed}s"
        echo ""
        info "Last 20 lines:"
        tail -20 "$SCRIPT_DIR/last_run.log"
        return 1
    fi
}

# ─── Validate data ───────────────────────────────────────────────────────────

validate_data() {
    banner "Validating Physics"

    if [ ! -f "$DATA_DIR/GranularTemp.txt" ]; then
        fail "No GranularTemp.txt found in $DATA_DIR"
        return 1
    fi

    local lines=$(wc -l < "$DATA_DIR/GranularTemp.txt")
    info "Data: $lines data points in $DATA_DIR/GranularTemp.txt"

    echo ""
    echo "  First 3 data points:"
    head -3 "$DATA_DIR/GranularTemp.txt" | while read line; do echo "    $line"; done
    echo "  Last 3 data points:"
    tail -3 "$DATA_DIR/GranularTemp.txt" | while read line; do echo "    $line"; done
    echo ""

    # Run the validation script (it expects data/ in CWD)
    cd "$SCRIPT_DIR"
    python3 -c "
import numpy as np, sys

data = np.loadtxt('$DATA_DIR/GranularTemp.txt')
if data.ndim == 1:
    data = data.reshape(1, -1)

steps = data[:, 0]
temps = data[:, 2]

print('=' * 50)
print('Granular Gas Physics Validation')
print('=' * 50)

passed = 0
total = 0

# 1. No NaN/Inf
total += 1
if np.all(np.isfinite(temps)):
    print('  No NaN/Inf:        PASS')
    passed += 1
else:
    print('  No NaN/Inf:        FAIL')

# 2. All temps >= 0
total += 1
if np.all(temps >= 0):
    print('  Non-negative T:    PASS')
    passed += 1
else:
    print(f'  Non-negative T:    FAIL (min={np.min(temps):.6e})')

# 3. Monotonic cooling (no spikes > 10% above previous)
total += 1
if len(temps) > 2:
    ratios = temps[1:] / temps[:-1]
    max_spike = np.max(ratios)
    if max_spike < 1.10:
        print(f'  No energy spikes:  PASS (max ratio={max_spike:.4f})')
        passed += 1
    else:
        bad_idx = np.argmax(ratios)
        print(f'  No energy spikes:  FAIL (spike at step {steps[bad_idx+1]:.0f}, ratio={max_spike:.4f})')
else:
    print('  No energy spikes:  SKIP')

# 4. Final T < initial T
total += 1
T_init = temps[0]
T_final = temps[-1]
if T_init > 0 and T_final < T_init:
    ratio = T_final / T_init
    print(f'  Cooling (Tf<Ti):   PASS (Tf/Ti={ratio:.4f})')
    passed += 1
else:
    print(f'  Cooling (Tf<Ti):   FAIL (Ti={T_init:.6e}, Tf={T_final:.6e})')

# 5. Cooling trend
total += 1
if len(steps) > 2:
    coeffs = np.polyfit(steps[1:], temps[1:], 1)
    if coeffs[0] < 0:
        print(f'  Cooling slope:     PASS (slope={coeffs[0]:.6e})')
        passed += 1
    else:
        print(f'  Cooling slope:     FAIL (slope={coeffs[0]:.6e})')

print(f'\nResults: {passed}/{total} checks passed')
if passed == total:
    print('ALL CHECKS PASSED')
    sys.exit(0)
else:
    print(f'WARNING: {total - passed} check(s) failed')
    sys.exit(1)
"
}

# ─── Haff analysis + plot ────────────────────────────────────────────────────

haff_plot() {
    banner "Haff's Law Analysis"

    if [ ! -f "$DATA_DIR/GranularTemp.txt" ]; then
        fail "No data to analyze"
        return 1
    fi

    cd "$SCRIPT_DIR"
    python3 haff_analysis.py "$DATA_DIR/GranularTemp.txt" 2>&1

    if [ -f haff_comparison.png ]; then
        pass "Plot saved: $SCRIPT_DIR/haff_comparison.png"
    fi
}

# ─── Main ─────────────────────────────────────────────────────────────────────

case "$MODE" in
    quick)
        banner "DEM Physics Test — Quick (100k steps)"
        build_sim
        run_sim "$SCRIPT_DIR/run_100k_test.toml" "100k step test (thermo every 100)"
        validate_data
        ;;
    full)
        banner "DEM Physics Test — Full (2M steps)"
        build_sim
        run_sim "$SCRIPT_DIR/run_2m_single.toml" "2M step Haff validation (thermo every 500)"
        validate_data
        haff_plot
        ;;
    check)
        banner "DEM Physics Test — Check Existing Data"
        validate_data
        if command -v python3 &>/dev/null && python3 -c "import matplotlib" 2>/dev/null; then
            haff_plot
        else
            info "Skipping Haff plot (matplotlib not available)"
        fi
        ;;
    *)
        echo "Usage: $0 [quick|full|check]"
        echo "  quick  — Build + run 100k steps + validate (default)"
        echo "  full   — Build + run 2M steps + validate + Haff plot"
        echo "  check  — Validate + plot existing data (no simulation)"
        exit 1
        ;;
esac

echo ""
banner "Done"

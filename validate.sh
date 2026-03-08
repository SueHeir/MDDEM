#!/usr/bin/env bash
# MDDEM Validation Script
# Runs cargo tests, builds examples, and validates physics output.
#
# Usage:
#   ./validate.sh              # auto-detect changed categories, short mode
#   ./validate.sh --long       # auto-detect, full production configs
#   ./validate.sh --dem        # only DEM examples
#   ./validate.sh --md         # only MD examples
#   ./validate.sh --all        # everything
#   ./validate.sh --all --long # everything, production step counts

set -euo pipefail
cd "$(dirname "$0")"

# --- Parse arguments ---
MODE="short"
FLAG_DEM=false
FLAG_MD=false
FLAG_CORE=false
FLAG_ALL=false
FLAG_EXPLICIT=false

for arg in "$@"; do
    case "$arg" in
        --long)  MODE="long" ;;
        --short) MODE="short" ;;
        --dem)   FLAG_DEM=true; FLAG_EXPLICIT=true ;;
        --md)    FLAG_MD=true; FLAG_EXPLICIT=true ;;
        --core)  FLAG_CORE=true; FLAG_EXPLICIT=true ;;
        --all)   FLAG_ALL=true; FLAG_EXPLICIT=true ;;
        *)       echo "Unknown option: $arg"; exit 1 ;;
    esac
done

if $FLAG_ALL || $FLAG_CORE; then
    FLAG_DEM=true
    FLAG_MD=true
fi

# --- Auto-detect from git diff if no explicit flags ---
if ! $FLAG_EXPLICIT; then
    echo "Auto-detecting changed categories..."
    changed_files=$(git diff --name-only HEAD 2>/dev/null || true)
    if [ -z "$changed_files" ]; then
        echo "No changes detected, running all categories."
        FLAG_DEM=true
        FLAG_MD=true
    else
        while IFS= read -r file; do
            case "$file" in
                crates/dem_*|crates/dem_*/*)   FLAG_DEM=true ;;
                crates/md_*|crates/md_*/*)     FLAG_MD=true ;;
                crates/mddem*|Cargo.*)         FLAG_DEM=true; FLAG_MD=true ;;
                examples/lj_*|examples/lj_*/*) FLAG_MD=true ;;
                examples/*|examples/*/*)        FLAG_DEM=true ;;
            esac
        done <<< "$changed_files"
    fi
    # If still nothing matched, run all
    if ! $FLAG_DEM && ! $FLAG_MD; then
        FLAG_DEM=true
        FLAG_MD=true
    fi
fi

echo "=== MDDEM Validation ($MODE mode) ==="
echo "  DEM: $FLAG_DEM"
echo "  MD:  $FLAG_MD"
echo ""

# --- Tracking ---
PASS=0
FAIL=0
SKIP=0
RESULTS=()

record() {
    local name="$1" status="$2"
    RESULTS+=("$status $name")
    case "$status" in
        PASS) PASS=$((PASS + 1)) ;;
        FAIL) FAIL=$((FAIL + 1)) ;;
        SKIP*) SKIP=$((SKIP + 1)) ;;
    esac
}

# --- Timeout wrapper (macOS compatible) ---
TIMEOUT=120
if [ "$MODE" = "long" ]; then
    TIMEOUT=0
fi

run_with_timeout() {
    local secs="$1"; shift
    if [ "$secs" -le 0 ]; then
        "$@"
        return $?
    fi
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    if wait "$pid" 2>/dev/null; then
        kill "$watcher" 2>/dev/null
        wait "$watcher" 2>/dev/null
        return 0
    else
        kill "$watcher" 2>/dev/null
        wait "$watcher" 2>/dev/null
        return 1
    fi
}

# --- Helper to run an example with optional MPI and physics validation ---
# run_example NAME CONFIG CATEGORY [NPROCS]
run_example() {
    local name="$1"
    local config="$2"
    local category="$3"
    local nprocs="${4:-1}"

    local label="$name"
    if [ "$nprocs" -gt 1 ]; then
        label="$name (MPI ${nprocs}p)"
    fi

    # Skip based on category flags
    case "$category" in
        DEM) if ! $FLAG_DEM; then record "$label" "SKIP"; echo "  $label ... SKIP"; return; fi ;;
        MD)  if ! $FLAG_MD;  then record "$label" "SKIP"; echo "  $label ... SKIP"; return; fi ;;
    esac

    echo -n "  $label ... "

    # Run simulation
    local sim_ok=true
    if [ "$nprocs" -gt 1 ]; then
        if ! command -v mpirun &>/dev/null; then
            record "$label" "SKIP (no mpirun)"
            echo "SKIP (no mpirun)"
            return
        fi
        if ! run_with_timeout "$TIMEOUT" mpirun --oversubscribe -np "$nprocs" cargo run --release --example "$name" -- "$config" > /dev/null 2>&1; then
            sim_ok=false
        fi
    else
        if ! run_with_timeout "$TIMEOUT" cargo run --release --example "$name" -- "$config" > /dev/null 2>&1; then
            sim_ok=false
        fi
    fi

    if ! $sim_ok; then
        record "$label" "FAIL"
        echo "FAIL (simulation crashed)"
        return
    fi

    # Run physics validation if validate.py exists
    if [ -f "examples/$name/validate.py" ]; then
        if python3 "examples/$name/validate.py" > /dev/null 2>&1; then
            record "$label" "PASS"
            echo "PASS"
        else
            record "$label" "FAIL"
            echo "FAIL (physics validation)"
        fi
    else
        # No validate.py — crash check only
        record "$label" "PASS"
        echo "PASS"
    fi
}

# --- Step 1: cargo test ---
echo "--- cargo test --workspace ---"
if cargo test --workspace 2>&1; then
    record "cargo test" "PASS"
    echo "  -> PASS"
else
    record "cargo test" "FAIL"
    echo "  -> FAIL"
fi
echo ""

# --- Step 2: Build all examples in release mode (MPI enabled) ---
echo "--- Building examples (release) ---"
if cargo build --release --examples 2>&1; then
    record "cargo build" "PASS"
    echo "  -> PASS"
else
    record "cargo build" "FAIL"
    echo "  -> FAIL"
    echo "Build failed, skipping example runs."
    FLAG_DEM=false
    FLAG_MD=false
fi
echo ""

# --- Step 3: DEM examples ---
echo "--- DEM Examples ---"
if [ "$MODE" = "long" ]; then
    run_example "granular_gas_benchmark" "examples/granular_gas_benchmark/validate_long_config.toml"     "DEM"
    run_example "granular_gas_benchmark" "examples/granular_gas_benchmark/validate_long_mpi_config.toml" "DEM" 4
    run_example "hopper"                 "examples/hopper/validate_long_config.toml"                      "DEM"
else
    run_example "granular_gas_benchmark" "examples/granular_gas_benchmark/validate_config.toml"     "DEM"
    run_example "granular_gas_benchmark" "examples/granular_gas_benchmark/validate_mpi_config.toml" "DEM" 4
    run_example "hopper"                 "examples/hopper/validate_config.toml"                      "DEM"
fi
run_example "toml_single" "" "DEM"
echo ""

# --- Step 4: MD examples ---
echo "--- MD Examples ---"
if [ "$MODE" = "long" ]; then
    run_example "lj_argon" "examples/lj_argon/validate_long_config.toml"     "MD"
    run_example "lj_argon" "examples/lj_argon/validate_long_mpi_config.toml" "MD" 4
else
    run_example "lj_argon" "examples/lj_argon/validate_config.toml"     "MD"
    run_example "lj_argon" "examples/lj_argon/validate_mpi_config.toml" "MD" 4
fi
echo ""

# --- Summary ---
echo "========================================="
echo "  PASS: $PASS   FAIL: $FAIL   SKIP: $SKIP"
echo "========================================="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "VALIDATION FAILED"
    exit 1
else
    echo "VALIDATION PASSED"
    exit 0
fi

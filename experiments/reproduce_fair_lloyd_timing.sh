#!/usr/bin/env bash
# =============================================================================
# reproduce_fair_lloyd_timing.sh
#
# Reproduce the "Computational cost of Fair Lloyd's algorithm" experiment
# (Table: fairLloydTiming) from:
#   "A scalable linear programming-based framework for data clustering"
#   Khajavirad, Shen, Wang (2026) — Section 4.2.
#
# Runs Fair Lloyd's algorithm in heuristic_only mode on the three largest
# datasets (Credit, Adult2000, Adult3000) for K ∈ {2,3,4,5}, ρ ∈ {0.99,0.90},
# both α-fair and τ-fair constraints, 5 random seeds each.
#
# Produces two CSV files in the output directory:
#   fair_lloyd_raw.csv     — per-run wall-clock time, iterations, objective
#   fair_lloyd_summary.csv — mean (std) aggregated for the paper table
#
# Usage
# -----
#   chmod +x reproduce_fair_lloyd_timing.sh
#   ./reproduce_fair_lloyd_timing.sh -d /path/to/FairClusteringData
#
#   ./reproduce_fair_lloyd_timing.sh -d ./FairClusteringData -e ./build/cutLPK -o ./timing_results
#
#   # Dry-run (print commands without executing):
#   ./reproduce_fair_lloyd_timing.sh -d ./FairClusteringData --dry-run
#
# Options
# -------
#   -d, --data-dir DIR     Path to FairClusteringData folder (required)
#   -e, --exe PATH         Path to cutLPK binary (auto-detected if omitted)
#   -o, --out-dir DIR      Output directory (default: ./fair_lloyd_timing_results)
#   -v, --verbose          Show cutLPK console output (default: silent)
#   --dry-run              Print commands without executing
#   -h, --help             Show this help message
#
# Requirements
# ------------
#   - Compiled cutLPK binary with Gurobi support (ENABLE_GUROBI=ON)
#   - cuPDLPx + Gurobi shared libraries in PATH / LD_LIBRARY_PATH
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paper configuration (Table: fairLloydTiming)
# ---------------------------------------------------------------------------
KS=(2 3 4 5)
RHOS=(0.99 0.90)
FAIR_TYPES=("alpha" "tau")
SEEDS=(0 1 2 3 4)

# Datasets for the timing table (the three largest)
declare -A DATA_FILE=(
    [Credit]="Credit_data.csv"
    [AS1]="adult2000.csv"
    [AS2]="adult3000.csv"
)
declare -A LABEL_FILE=(
    [Credit]="Credit_labels.csv"
    [AS1]="adult2000_labels.csv"
    [AS2]="adult3000_labels.csv"
)
DATASETS=(Credit AS1 AS2)

# Fair Lloyd parameters (heuristic_only mode — no cutting plane)
FAIR_ASSIGNMENT_SOLVER="gurobi"

# ---- Auto-detect defaults ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_EXE=""
for cand in \
    "${SCRIPT_DIR}/../build-msvc/Release/cutLPK.exe" \
    "${SCRIPT_DIR}/../build-msvc/Release/cutLPK" \
    "${SCRIPT_DIR}/../build/cutLPK"; do
    if [ -x "$cand" ] || [ -f "$cand" ]; then
        DEFAULT_EXE="$cand"
        break
    fi
done

# ---- CLI argument parsing ----
DATA_DIR=""
EXE="${DEFAULT_EXE}"
OUT_DIR=""
DRY_RUN=false
VERBOSE=false

usage() {
    head -33 "$0" | tail -14
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data-dir)   DATA_DIR="$2";   shift 2 ;;
        -e|--exe)        EXE="$2";        shift 2 ;;
        -o|--out-dir)    OUT_DIR="$2";    shift 2 ;;        -v|--verbose)    VERBOSE=true;     shift   ;;        --dry-run)       DRY_RUN=true;    shift   ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$DATA_DIR" ]; then
    echo "ERROR: --data-dir is required."
    usage
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

if [ -z "$EXE" ] || [ ! -f "$EXE" ]; then
    echo "ERROR: cutLPK binary not found. Build the project or pass --exe."
    exit 1
fi
EXE="$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")"

if [ -z "$OUT_DIR" ]; then
    OUT_DIR="${SCRIPT_DIR}/fair_lloyd_timing_results"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# ---- Count total runs ----
NDS="${#DATASETS[@]}"
NKS="${#KS[@]}"
NRHOS="${#RHOS[@]}"
NFT="${#FAIR_TYPES[@]}"
NSEEDS="${#SEEDS[@]}"
TOTAL=$((NDS * NKS * NRHOS * NFT * NSEEDS))

echo "=============================================================================="
echo " Fair Lloyd Timing Experiment"
echo "=============================================================================="
echo " Datasets : ${DATASETS[*]}"
echo " K values : ${KS[*]}"
echo " Rho      : ${RHOS[*]}"
echo " Fairness : ${FAIR_TYPES[*]}"
echo " Seeds    : ${SEEDS[*]}  (${NSEEDS} per combo)"
echo " Total runs: ${TOTAL}"
echo " Binary   : ${EXE}"
echo " Data dir : ${DATA_DIR}"
echo " Output   : ${OUT_DIR}"
if [ "$DRY_RUN" = true ]; then
    echo " Mode     : DRY RUN (no execution)"
fi
echo "=============================================================================="
echo ""

# ---- Output CSV paths ----
RAW_CSV="${OUT_DIR}/fair_lloyd_raw.csv"
SUMMARY_CSV="${OUT_DIR}/fair_lloyd_summary.csv"

# ---- Add exe directory to library search path ----
EXE_DIR="$(dirname "$EXE")"
export PATH="${EXE_DIR}:${PATH}"
export LD_LIBRARY_PATH="${EXE_DIR}:${LD_LIBRARY_PATH:-}"

# ---- Helper: compute mean and std from a list of numbers ----
calc_stats() {
    # Reads numbers from stdin, prints "mean std"
    awk '
    {
        sum += $1
        sumsq += $1 * $1
        n++
    }
    END {
        if (n == 0) { printf "N/A N/A"; exit }
        mean = sum / n
        if (n > 1)
            std = sqrt((sumsq - sum * sum / n) / (n - 1))
        else
            std = 0.0
        printf "%.1f %.1f", mean, std
    }'
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
run=0
warnings=0
WALL_START=$(date +%s)

# Initialise raw CSV
if [ "$DRY_RUN" = false ]; then
    echo "dataset,K,fairness_type,rho,seed,time_sec,iterations,objective" > "$RAW_CSV"
fi

for ds in "${DATASETS[@]}"; do
    data_f="${DATA_DIR}/${DATA_FILE[$ds]}"
    label_f="${DATA_DIR}/${LABEL_FILE[$ds]}"

    if [ ! -f "$data_f" ]; then
        echo "ERROR: Data file not found: $data_f"
        warnings=$((warnings + 1))
        continue
    fi
    if [ ! -f "$label_f" ]; then
        echo "ERROR: Label file not found: $label_f"
        warnings=$((warnings + 1))
        continue
    fi

    for K in "${KS[@]}"; do
        for rho in "${RHOS[@]}"; do
            for fair in "${FAIR_TYPES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    run=$((run + 1))

                    # fairness_param: rho for alpha, rho/K for tau
                    if [ "$fair" = "alpha" ]; then
                        fair_param="$rho"
                    else
                        fair_param=$(awk "BEGIN {printf \"%.6f\", ${rho}/${K}}")
                    fi

                    rho_tag="${rho/./p}"
                    tag="${ds}_K${K}_${fair}_rho${rho_tag}_seed${seed}"
                    log_file="${OUT_DIR}/tmp_${tag}.log"
                    out_file="${OUT_DIR}/tmp_${tag}_out.txt"

                    cmd=(
                        "${EXE}"
                        "${data_f}"
                        "${K}"
                        "heuristic_only=true"
                        "fairness_type=${fair}"
                        "fairness_param=${fair_param}"
                        "group_file=${label_f}"
                        "lloyd_random_starts=1"
                        "random_seed=${seed}"
                        "output_file=${out_file}"
                        "fair_assignment_solver=${FAIR_ASSIGNMENT_SOLVER}"
                        "cutting_plane_verbose=1"
                        "output_level=1"
                    )

                    tag_msg="[${run}/${TOTAL}] ${ds} K=${K} ${fair} rho=${rho} seed=${seed}"

                    if [ "$DRY_RUN" = true ]; then
                        echo "${tag_msg}"
                        echo "  CMD: ${cmd[*]}"
                        continue
                    fi

                    printf "%s ... " "${tag_msg}"

                    t0=$(date +%s%3N)  # milliseconds
                    if [ "$VERBOSE" = true ]; then
                        "${cmd[@]}" 2>&1 | tee "${log_file}" && rc=0 || rc=$?
                    else
                        "${cmd[@]}" > "${log_file}" 2>&1 && rc=0 || rc=$?
                    fi
                    t1=$(date +%s%3N)
                    elapsed=$(awk "BEGIN {printf \"%.3f\", (${t1}-${t0})/1000}")

                    # Parse iteration count from log
                    iter_count=-1
                    if iter_line=$(grep -oP 'converged after \K\d+' "$log_file" 2>/dev/null | head -1); then
                        iter_count="$iter_line"
                    fi

                    # Parse objective from log
                    objective="N/A"
                    if obj_line=$(grep -oP 'Fair Lloyd upper bound:\s*\K[0-9]+\.?[0-9]*' "$log_file" 2>/dev/null | head -1); then
                        objective="$obj_line"
                    fi

                    if [ "$iter_count" -lt 0 ]; then
                        echo "WARN: parse failed  ${elapsed}s"
                        warnings=$((warnings + 1))
                    else
                        echo "${elapsed}s, ${iter_count} iters, obj=${objective}"
                    fi

                    # Append to raw CSV
                    echo "${ds},${K},${fair},${rho},${seed},${elapsed},${iter_count},${objective}" >> "$RAW_CSV"

                    # Clean up temp files
                    rm -f "$log_file" "$out_file"
                done
            done
        done
    done
done

WALL_END=$(date +%s)
WALL_TOTAL=$((WALL_END - WALL_START))

# ---------------------------------------------------------------------------
# Build summary CSV matching paper Table "fairLloydTiming"
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" = false ]; then
    echo "dataset,K,rho,alpha_time_mean,alpha_time_std,alpha_iter_mean,alpha_iter_std,tau_time_mean,tau_time_std,tau_iter_mean,tau_iter_std" > "$SUMMARY_CSV"

    for ds in "${DATASETS[@]}"; do
        for K in "${KS[@]}"; do
            for rho in "${RHOS[@]}"; do
                rho_str="$(printf '%.2f' "$rho")"

                # Extract alpha rows: time_sec (col 6) and iterations (col 7)
                alpha_times=$(awk -F',' -v ds="$ds" -v K="$K" -v r="$rho_str" \
                    '$1==ds && $2==K && $3=="alpha" && $4==r && $7>=0 {print $6}' "$RAW_CSV")
                alpha_iters=$(awk -F',' -v ds="$ds" -v K="$K" -v r="$rho_str" \
                    '$1==ds && $2==K && $3=="alpha" && $4==r && $7>=0 {print $7}' "$RAW_CSV")

                # Extract tau rows
                tau_times=$(awk -F',' -v ds="$ds" -v K="$K" -v r="$rho_str" \
                    '$1==ds && $2==K && $3=="tau" && $4==r && $7>=0 {print $6}' "$RAW_CSV")
                tau_iters=$(awk -F',' -v ds="$ds" -v K="$K" -v r="$rho_str" \
                    '$1==ds && $2==K && $3=="tau" && $4==r && $7>=0 {print $7}' "$RAW_CSV")

                read -r aTm aTs <<< "$(echo "$alpha_times" | calc_stats)"
                read -r aIm aIs <<< "$(echo "$alpha_iters" | calc_stats)"
                read -r tTm tTs <<< "$(echo "$tau_times" | calc_stats)"
                read -r tIm tIs <<< "$(echo "$tau_iters" | calc_stats)"

                echo "${ds},${K},${rho},${aTm},${aTs},${aIm},${aIs},${tTm},${tTs},${tIm},${tIs}" >> "$SUMMARY_CSV"
            done
        done
    done
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=============================================================================="
printf " Done in %dm %ds.\n" $((WALL_TOTAL/60)) $((WALL_TOTAL%60))
echo " Raw data     : ${RAW_CSV}"
echo " Summary table: ${SUMMARY_CSV}"
if [ "$warnings" -gt 0 ]; then
    echo " Warnings     : ${warnings} (check logs above)"
fi
echo "=============================================================================="

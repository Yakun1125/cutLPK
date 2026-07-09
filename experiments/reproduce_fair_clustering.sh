#!/usr/bin/env bash
# =============================================================================
# reproduce_fair_clustering.sh
#
# Reproduce the fair K-means clustering experiments from:
#   "A scalable linear programming-based framework for data clustering"
#   Khajavirad, Shen, Wang (2026) — Section 4, Tables 1–3.
#
# Runs the cutLPK cutting-plane algorithm on all 10 real-world datasets for
# both alpha-fair (proportional representation) and tau-fair (minimum group
# fraction) constraints, matching exactly the 286 instances reported in the
# paper's longtables.
#
# Usage
# -----
#   chmod +x reproduce_fair_clustering.sh
#   ./reproduce_fair_clustering.sh -d /path/to/FairClusteringData
#
#   ./reproduce_fair_clustering.sh -d ./FairClusteringData -e ./build/cutLPK -o ./results
#
#   # Dry-run (print commands without executing):
#   ./reproduce_fair_clustering.sh -d ./FairClusteringData --dry-run
#
#   # Resume from job N:
#   ./reproduce_fair_clustering.sh -d ./FairClusteringData -s 150
#
# Options
# -------
#   -d, --data-dir DIR     Path to FairClusteringData folder (required)
#   -e, --exe PATH         Path to cutLPK binary (auto-detected if omitted)
#   -o, --out-dir DIR      Output directory (default: DATA_DIR/../results/fair)
#   -s, --start-from N     1-based job index to resume from (default: 1)
#   -S, --seed N           Random seed (default: 42)
#   -v, --verbose          Show cutLPK console output (default: silent)
#   --dry-run              Print commands without executing
#   -h, --help             Show this help message
#
# Requirements
# ------------
#   - Compiled cutLPK binary
#   - cuPDLPx + HiGHS shared libraries in PATH / LD_LIBRARY_PATH
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paper's cutting-plane parameters for fair clustering (Section 4.2)
# ---------------------------------------------------------------------------
MAX_CUTS_INIT=1000000          # p_init
NUM_ITER_NO_IMPROVE=5          # 5 consecutive iters w/o gap improvement
OPT_GAP="1e-4"                 # epsilon_opt
TIME_LIMIT_ALL=10800           # 3 hours (seconds)
MAX_SEPARATION_TIME=300        # 300 seconds in one separation call
CUTS_VIO_TOL="1e-4"
SOLVER="cupdlpx"
LLOYD_RANDOM_STARTS=100
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
START_FROM=1
SEED=42
DRY_RUN=false
VERBOSE=false

usage() {
    head -36 "$0" | tail -16
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data-dir)   DATA_DIR="$2";   shift 2 ;;
        -e|--exe)        EXE="$2";        shift 2 ;;
        -o|--out-dir)    OUT_DIR="$2";    shift 2 ;;
        -s|--start-from) START_FROM="$2"; shift 2 ;;
        -S|--seed)       SEED="$2";       shift 2 ;;        -v|--verbose)    VERBOSE=true;     shift   ;;        --dry-run)       DRY_RUN=true;    shift   ;;
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
    OUT_DIR="${DATA_DIR}/../results/fair"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# ---- Dataset → (data_file, label_file) map ----
declare -A DATA_FILE=(
    [HH]="HH_data.csv"       [HC]="HC_data.csv"
    [SM]="Math_data.csv"     [WDBC]="WDBC_data.csv"
    [SP]="Port_data.csv"     [Titanic2]="Titanic_data.csv"
    [Titanic3]="Titanic3_data.csv"  [Credit]="Credit_data.csv"
    [AS1]="adult2000.csv"    [AS2]="adult3000.csv"
)
declare -A LABEL_FILE=(
    [HH]="HH_labels.csv"       [HC]="HC_labels.csv"
    [SM]="Math_labels.csv"     [WDBC]="WDBC_labels.csv"
    [SP]="Port_labels.csv"     [Titanic2]="Titanic_labels.csv"
    [Titanic3]="Titanic3_labels.csv"  [Credit]="Credit_labels.csv"
    [AS1]="adult2000_labels.csv"    [AS2]="adult3000_labels.csv"
)

# ---- Dataset display names ----
declare -A DISPLAY=(
    [HH]="HH" [HC]="HC" [SM]="SM" [WDBC]="WDBC" [SP]="SP"
    [Titanic2]="Titanic 2" [Titanic3]="Titanic 3"
    [Credit]="Credit" [AS1]="AS1" [AS2]="AS2"
)

# ---- Paper experiment combos: dataset_key → list of "K:rho" ----
# These match exactly the rows present in the paper's longtables.
declare -A COMBOS
COMBOS[HH]="2:0.99 2:0.9 2:0.8 3:0.99 3:0.9 3:0.8 4:0.99 4:0.9 4:0.8 4:0.7 5:0.99 5:0.9 5:0.8 5:0.7"
COMBOS[HC]="2:0.99 2:0.9 2:0.8 2:0.7 3:0.99 3:0.9 3:0.8 3:0.7 4:0.99 4:0.9 4:0.8 4:0.7 5:0.99 5:0.9 5:0.8 5:0.7"
COMBOS[SM]="2:0.99 3:0.99 3:0.9 3:0.8 3:0.7 4:0.99 4:0.9 4:0.8 4:0.7 5:0.99 5:0.9 5:0.8 5:0.7"
COMBOS[WDBC]="2:0.99 2:0.9 2:0.8 2:0.7 3:0.99 3:0.9 3:0.8 3:0.7 4:0.99 4:0.9 4:0.8 4:0.7 5:0.99 5:0.9 5:0.8 5:0.7"
COMBOS[SP]="2:0.99 2:0.9 3:0.99 3:0.9 3:0.8 3:0.7 4:0.99 4:0.9 4:0.8 4:0.7 5:0.99 5:0.9 5:0.8 5:0.7"
COMBOS[Titanic2]="2:0.99 2:0.9 2:0.8 2:0.7 3:0.99 3:0.9 3:0.8 4:0.99 4:0.9 4:0.8 4:0.7 5:0.99 5:0.9 5:0.8 5:0.7"
COMBOS[Titanic3]="2:0.99 2:0.9 2:0.8 2:0.7 3:0.99 3:0.9 3:0.8 3:0.7 4:0.99 4:0.9 4:0.8 4:0.7 5:0.99 5:0.9 5:0.8 5:0.7"
COMBOS[Credit]="2:0.99 2:0.9 2:0.8 2:0.7 3:0.99 3:0.9 3:0.8 3:0.7 4:0.99 4:0.9 4:0.8 4:0.7 5:0.99 5:0.9 5:0.8 5:0.7"
COMBOS[AS1]="2:0.99 2:0.9 2:0.8 3:0.99 3:0.9 4:0.99 4:0.9 4:0.8 4:0.7 5:0.99 5:0.9 5:0.8 5:0.7"
COMBOS[AS2]="2:0.99 2:0.9 3:0.99 3:0.9 4:0.99 4:0.9 4:0.8 5:0.99 5:0.9 5:0.8"

FAIR_TYPES=("alpha" "tau")
DATASETS=(HH HC SM WDBC SP Titanic2 Titanic3 Credit AS1 AS2)

# ---- Build flat job list ----
JOBS=()
for ds in "${DATASETS[@]}"; do
    data_f="${DATA_DIR}/${DATA_FILE[$ds]}"
    label_f="${DATA_DIR}/${LABEL_FILE[$ds]}"
    for fair in "${FAIR_TYPES[@]}"; do
        for combo in ${COMBOS[$ds]}; do
            K="${combo%%:*}"
            rho="${combo##*:}"
            JOBS+=("${ds}|${DISPLAY[$ds]}|${data_f}|${label_f}|${K}|${rho}|${fair}")
        done
    done
done

TOTAL="${#JOBS[@]}"

echo "=============================================================================="
echo " Fair Clustering Reproduce Script"
echo "=============================================================================="
echo " Total jobs     : ${TOTAL}  (paper reports 286 instances)"
echo " Datasets       : ${DATASETS[*]}"
echo " Fairness types : ${FAIR_TYPES[*]}"
echo " Binary         : ${EXE}"
echo " Data dir       : ${DATA_DIR}"
echo " Output dir     : ${OUT_DIR}"
echo " Random seed    : ${SEED}"
echo " Verbose        : ${VERBOSE}"
echo " Start from     : ${START_FROM}"
if [ "$DRY_RUN" = true ]; then
    echo " Mode           : DRY RUN (no execution)"
fi
echo "=============================================================================="
echo ""

# ---- Summary CSV ----
SUMMARY_CSV="${OUT_DIR}/fair_clustering_summary.csv"
if [ "$DRY_RUN" = false ] && [ "$START_FROM" -le 1 ]; then
    echo "dataset,K,fairness_type,rho,fairness_param,wall_time_sec,exit_code" > "$SUMMARY_CSV"
fi

# ---- Add exe directory to library search path ----
EXE_DIR="$(dirname "$EXE")"
export PATH="${EXE_DIR}:${PATH}"
export LD_LIBRARY_PATH="${EXE_DIR}:${LD_LIBRARY_PATH:-}"

# ---- Run ----
SUCCESS=0
FAIL=0
WALL_START=$(date +%s)

idx=0
for job in "${JOBS[@]}"; do
    idx=$((idx + 1))
    if [ "$idx" -lt "$START_FROM" ]; then
        continue
    fi

    IFS='|' read -r ds_key ds_display data_f label_f K rho fair <<< "$job"

    # fairness_param: rho for alpha, rho/K for tau (matches paper eqs.)
    if [ "$fair" = "alpha" ]; then
        fair_param="$rho"
    else
        fair_param=$(awk "BEGIN {printf \"%.6f\", ${rho}/${K}}")
    fi

    # Output file stem
    rho_tag="${rho/./p}"
    stem="${ds_key}_K${K}_${fair}_rho${rho_tag}"
    output_file="${OUT_DIR}/${stem}_output.txt"

    # Build command
    cmd=(
        "${EXE}"
        "${data_f}"
        "${K}"
        "fairness_type=${fair}"
        "fairness_param=${fair_param}"
        "group_file=${label_f}"
        "output_file=${output_file}"
        "max_cuts_init=${MAX_CUTS_INIT}"
        "num_iter_no_improve=${NUM_ITER_NO_IMPROVE}"
        "opt_gap=${OPT_GAP}"
        "time_limit_all=${TIME_LIMIT_ALL}"
        "max_separation_time=${MAX_SEPARATION_TIME}"
        "cuts_vio_tol=${CUTS_VIO_TOL}"
        "solver=${SOLVER}"
        "lloyd_random_starts=${LLOYD_RANDOM_STARTS}"
        "fair_assignment_solver=${FAIR_ASSIGNMENT_SOLVER}"
        "random_seed=${SEED}"
    )

    tag="[${idx}/${TOTAL}] ${ds_display}  K=${K}  ${fair}  rho=${rho}"
    echo "${tag}  running ..."

    if [ "$DRY_RUN" = true ]; then
        echo "  CMD: ${cmd[*]}"
        continue
    fi

    t0=$(date +%s)
    if [ "$VERBOSE" = true ]; then
        "${cmd[@]}" && rc=0 || rc=$?
    else
        "${cmd[@]}" > /dev/null 2>&1 && rc=0 || rc=$?
    fi
    t1=$(date +%s)
    wall=$((t1 - t0))

    if [ "$rc" -eq 0 ]; then
        echo "${tag}  OK  ${wall}s"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "${tag}  FAIL(rc=${rc})  ${wall}s"
        FAIL=$((FAIL + 1))
    fi

    echo "${ds_display},${K},${fair},${rho},${fair_param},${wall},${rc}" >> "$SUMMARY_CSV"
done

WALL_END=$(date +%s)
WALL_TOTAL=$((WALL_END - WALL_START))
ATTEMPTED=$((TOTAL - START_FROM + 1))
if [ "$ATTEMPTED" -lt 0 ]; then ATTEMPTED=0; fi

echo ""
echo "=============================================================================="
echo " Done.  ${SUCCESS} succeeded, ${FAIL} failed out of ${ATTEMPTED} attempted."
printf " Total wall-clock: %dh %dm %ds\n" $((WALL_TOTAL/3600)) $((WALL_TOTAL%3600/60)) $((WALL_TOTAL%60))
echo " Summary: ${SUMMARY_CSV}"
echo "=============================================================================="

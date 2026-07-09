#!/usr/bin/env bash
# =============================================================================
# reproduce_spectral_clustering.sh
#
# Reproduce the spectral clustering / community detection experiments from:
#   "A scalable linear programming-based framework for data clustering"
#   Khajavirad, Shen, Wang (2026) — Section 5.2, Table "social_network".
#
# Runs the cutLPK cutting-plane algorithm on 16 real-world social / citation
# networks for K ∈ {2,…,10} (friendship: K ∈ {4,…,10}), matching the
# instances reported in the paper.
#
# The script expects a flat directory of Laplacian-matrix CSV files (one per
# graph).  File names must match the mapping below (see --help for the list).
#
# Usage
# -----
#   chmod +x reproduce_spectral_clustering.sh
#   ./reproduce_spectral_clustering.sh -d /path/to/laplacian_csvs
#
#   ./reproduce_spectral_clustering.sh -d ./laplacians -e ./build/cutLPK -o ./results
#
#   # Dry-run (print commands without executing):
#   ./reproduce_spectral_clustering.sh -d ./laplacians --dry-run
#
#   # Resume from job N:
#   ./reproduce_spectral_clustering.sh -d ./laplacians -s 50
#
# Options
# -------
#   -d, --data-dir DIR     Path to folder containing Laplacian CSV files (required)
#   -e, --exe PATH         Path to cutLPK binary (auto-detected if omitted)
#   -o, --out-dir DIR      Output directory (default: DATA_DIR/../results/spectral)
#   -s, --start-from N     1-based job index to resume from (default: 1)
#   -S, --seed N           Random seed (default: 42)
#   -v, --verbose          Show cutLPK console output (default: silent)
#   --dry-run              Print commands without executing
#   -h, --help             Show this help message and the required file list
#
# Requirements
# ------------
#   - Compiled cutLPK binary
#   - cuPDLPx shared libraries in PATH / LD_LIBRARY_PATH
#   - Laplacian CSV files (see --help for the full list)
#
# Data sources (see paper §5.2)
# -----------------------------
#   University of Michigan:  https://websites.umich.edu/~mejn/netdata/
#   Stanford SNAP:           https://snap.stanford.edu/data/index.html
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paper's cutting-plane parameters for spectral clustering (Section 5.2)
# ---------------------------------------------------------------------------
MAX_CUTS_INIT=10000000         # p_init = 10^7
OPT_GAP="1e-4"
TIME_LIMIT_ALL=10800           # 3 hours
MAX_SEPARATION_TIME=300
CUTS_VIO_TOL="1e-4"
SOLVER="cupdlpx"
LLOYD_RANDOM_STARTS=100

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
    echo "Usage: $0 -d <laplacian_dir> [options]"
    echo ""
    echo "Options:"
    echo "  -d, --data-dir DIR    Path to folder with Laplacian CSV files (required)"
    echo "  -e, --exe PATH        Path to cutLPK binary"
    echo "  -o, --out-dir DIR     Output directory"
    echo "  -s, --start-from N    Resume from job N (1-based, default: 1)"
    echo "  -S, --seed N          Random seed (default: 42)"
    echo "  --dry-run             Print commands, do not execute"
    echo "  -h, --help            Show this help"
    echo ""
    echo "Required Laplacian CSV files in DATA_DIR:"
    echo "  polbooks_L.csv                    football_L.csv"
    echo "  adjnoun_L.csv                     facebook_L.csv"
    echo "  friendship_L.csv                  deezer_ego_nets_full_L.csv"
    echo "  netscience_giant_L.csv            ca-GrQc_sample500_L.csv"
    echo "  ca-GrQc_sample1000_L.csv          ca-GrQc_sample1500_L.csv"
    echo "  ca-HepTh_sample500_L.csv          ca-HepTh_sample1000_L.csv"
    echo "  ca-HepTh_sample1500_L.csv         lastfm_asia_sample500_L.csv"
    echo "  lastfm_asia_sample1000_L.csv      lastfm_asia_sample1500_L.csv"
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
    OUT_DIR="${DATA_DIR}/../results/spectral"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# ---- Dataset → Laplacian CSV filename map ----
# Keys match the paper's table.  Values must exist as files in DATA_DIR.
declare -A LAP_FILE=(
    [polbooks]="polbooks_L.csv"
    [football]="football_L.csv"
    [adjnoun]="adjnoun_L.csv"
    [facebook]="facebook_L.csv"
    [friendship]="friendship_L.csv"
    [deezer_ego]="deezer_ego_nets_full_L.csv"
    [netscience]="netscience_giant_L.csv"
    [ca-GrQc_500]="ca-GrQc_sample500_L.csv"
    [ca-GrQc_1000]="ca-GrQc_sample1000_L.csv"
    [ca-GrQc_1500]="ca-GrQc_sample1500_L.csv"
    [ca-HepTh_500]="ca-HepTh_sample500_L.csv"
    [ca-HepTh_1000]="ca-HepTh_sample1000_L.csv"
    [ca-HepTh_1500]="ca-HepTh_sample1500_L.csv"
    [lastfm_asia_500]="lastfm_asia_sample500_L.csv"
    [lastfm_asia_1000]="lastfm_asia_sample1000_L.csv"
    [lastfm_asia_1500]="lastfm_asia_sample1500_L.csv"
)

# K values for each dataset (matching paper Table "social_network").
# friendship: K=2,3 are trivial (three connected components → obj=0).
declare -A K_VALUES
K_VALUES[polbooks]="2 3 4 5 6 7 8 9 10"
K_VALUES[football]="2 3 4 5 6 7 8 9 10"
K_VALUES[adjnoun]="2 3 4 5 6 7 8 9 10"
K_VALUES[facebook]="2 3 4 5 6 7 8 9 10"
K_VALUES[friendship]="4 5 6 7 8 9 10"
K_VALUES[deezer_ego]="2 3 4 5 6 7 8 9 10"
K_VALUES[netscience]="2 3 4 5 6 7 8 9 10"
K_VALUES[ca-GrQc_500]="2 3 4 5 6 7 8 9 10"
K_VALUES[ca-GrQc_1000]="2 3 4 5 6 7 8 9 10"
K_VALUES[ca-GrQc_1500]="2 3 4 5 6 7 8 9 10"
K_VALUES[ca-HepTh_500]="2 3 4 5 6 7 8 9 10"
K_VALUES[ca-HepTh_1000]="2 3 4 5 6 7 8 9 10"
K_VALUES[ca-HepTh_1500]="2 3 4 5 6 7 8 9 10"
K_VALUES[lastfm_asia_500]="2 3 4 5 6 7 8 9 10"
K_VALUES[lastfm_asia_1000]="2 3 4 5 6 7 8 9 10"
K_VALUES[lastfm_asia_1500]="2 3 4 5 6 7 8 9 10"

DATASETS=(
    polbooks football adjnoun facebook friendship
    deezer_ego netscience
    ca-GrQc_500 ca-GrQc_1000 ca-GrQc_1500
    ca-HepTh_500 ca-HepTh_1000 ca-HepTh_1500
    lastfm_asia_500 lastfm_asia_1000 lastfm_asia_1500
)

# ---- Build flat job list ----
JOBS=()
for ds in "${DATASETS[@]}"; do
    lap_f="${DATA_DIR}/${LAP_FILE[$ds]}"
    for K in ${K_VALUES[$ds]}; do
        JOBS+=("${ds}|${lap_f}|${K}")
    done
done

TOTAL="${#JOBS[@]}"

echo "=============================================================================="
echo " Spectral Clustering Reproduce Script"
echo "=============================================================================="
echo " Total jobs     : ${TOTAL}"
echo " Datasets       : ${#DATASETS[@]}"
echo " K range        : 2..10  (friendship: 4..10)"
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
SUMMARY_CSV="${OUT_DIR}/spectral_clustering_summary.csv"
if [ "$DRY_RUN" = false ] && [ "$START_FROM" -le 1 ]; then
    echo "dataset,K,wall_time_sec,exit_code" > "$SUMMARY_CSV"
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

    IFS='|' read -r ds lap_f K <<< "$job"

    # Output file stem
    stem="${ds}_K${K}_spectral"
    output_file="${OUT_DIR}/${stem}_output.txt"

    cmd=(
        "${EXE}"
        "${lap_f}"
        "${K}"
        "is_spectral_clustering=true"
        "output_file=${output_file}"
        "max_cuts_init=${MAX_CUTS_INIT}"
        "opt_gap=${OPT_GAP}"
        "time_limit_all=${TIME_LIMIT_ALL}"
        "max_separation_time=${MAX_SEPARATION_TIME}"
        "cuts_vio_tol=${CUTS_VIO_TOL}"
        "solver=${SOLVER}"
        "lloyd_random_starts=${LLOYD_RANDOM_STARTS}"
        "random_seed=${SEED}"
    )

    tag="[${idx}/${TOTAL}] ${ds}  K=${K}"

    if [ "$DRY_RUN" = true ]; then
        echo "${tag}"
        echo "  CMD: ${cmd[*]}"
        continue
    fi

    # Check that the Laplacian file exists
    if [ ! -f "$lap_f" ]; then
        echo "${tag}  SKIP (file not found: ${lap_f})"
        FAIL=$((FAIL + 1))
        echo "${ds},${K},,," >> "$SUMMARY_CSV"
        continue
    fi

    printf "%s  running ... " "${tag}"

    t0=$(date +%s)
    if [ "$VERBOSE" = true ]; then
        "${cmd[@]}" && rc=0 || rc=$?
    else
        "${cmd[@]}" > /dev/null 2>&1 && rc=0 || rc=$?
    fi
    t1=$(date +%s)
    wall=$((t1 - t0))

    if [ "$rc" -eq 0 ]; then
        echo "OK  ${wall}s"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "FAIL(rc=${rc})  ${wall}s"
        FAIL=$((FAIL + 1))
    fi

    echo "${ds},${K},${wall},${rc}" >> "$SUMMARY_CSV"
done

WALL_END=$(date +%s)
WALL_TOTAL=$((WALL_END - WALL_START))
ATTEMPTED=$((TOTAL - START_FROM + 1))
if [ "$ATTEMPTED" -lt 0 ]; then ATTEMPTED=0; fi

echo ""
echo "=============================================================================="
echo " Done.  ${SUCCESS} succeeded, ${FAIL} failed/skipped out of ${ATTEMPTED} attempted."
printf " Total wall-clock: %dh %dm %ds\n" $((WALL_TOTAL/3600)) $((WALL_TOTAL%3600/60)) $((WALL_TOTAL%60))
echo " Summary: ${SUMMARY_CSV}"
echo "=============================================================================="

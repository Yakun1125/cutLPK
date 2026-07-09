<p align="center">
  <img src="cutLPK.png" alt="cutLPK Logo" width="300"/>
</p>

A scalable linear programming framework for solving **K-Means**, **Fair K-Means**, and **Spectral Clustering** problems to certified optimality. This project implements a cutting-plane algorithm built on the LP relaxation of the partition-matrix formulation. For algorithmic details and extensive computational results, see:

> A. Khajavirad, H. Shen, Y. Wang — *A scalable linear programming-based framework for data clustering* (2026).  
> [arXiv:2607.06709](https://arxiv.org/abs/2607.06709)

---

## Dependencies

| Dependency | Version | Required For | Notes |
|---|---|---|---|
| C++17 compiler | GCC ≥ 9 / MSVC ≥ 2019 / Clang ≥ 10 | Build | Linux or Windows; macOS is not supported (no NVIDIA GPU) |
| CMake | ≥ 3.16 | Build | |
| [cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx) | latest | LP solver (GPU) | Requires CUDA ≥ 12.4 and an NVIDIA GPU |
| [Gurobi](https://www.gurobi.com/) | ≥ 11.0 | Fair assignment | Free academic licenses available |
| [HiGHS](https://github.com/ERGO-Code/HiGHS) | ≥ 1.6 | Fair assignment | MIT-licensed alternative to Gurobi |
| [Eigen](http://eigen.tuxfamily.org/) | ≥ 3.4 | Linear algebra | Header-only |
| OpenMP | — | Multi-threaded separation | Usually bundled with compiler |

> **Note:** For paper-compatible results, build **with Gurobi enabled** (`ENABLE_GUROBI=ON`). The paper uses Gurobi as the fair assignment solver. Both Gurobi and HiGHS support both α-fair and τ-fair constraints.

---

## Prebuilt binary (Linux)

A precompiled, self-contained Linux binary is available for quick evaluation:

[**cutLPK prebuilt (Linux)**](https://drive.google.com/file/d/18m8l8Va-IVe8m2fCJtIDuep1Q9Ld5Dmu/view?usp=sharing)

Download, make it executable, and run directly — no additional dependency setup is needed.

> **Gurobi license:** If using `fair_assignment_solver=gurobi`, you need a valid license.  Place `gurobi.lic` in your home directory (`~/gurobi.lic`), or set the environment variable:\n>\n> ```bash\n> export GRB_LICENSE_FILE=/path/to/gurobi.lic\n> ```\n>\n> Free academic licenses are available at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

## Building

### 1. Prerequisites

Install and build **cuPDLPx** following [its own instructions](https://github.com/MIT-Lu-Lab/cuPDLPx).  Ensure CUDA ≥ 12.4 is available.

Install **Gurobi** (≥ 11.0), **HiGHS** (≥ 1.6), and **Eigen** (≥ 3.4) through your system package manager or from their official websites.

### 2. Configure

Edit `src/config.cmake` (or pass `-D` flags to CMake) to point to your dependency installations:

| CMake variable | Description |
|---|---|
| `EIGEN_INCLUDE_DIR` | Path to Eigen headers |
| `GUROBI_ROOT` | Path to Gurobi installation |
| `HIGHS_ROOT` | Path to HiGHS installation |
| `CUPDLPX_ROOT` | Path to cuPDLPx source |
| `CUPDLPX_BUILD_DIR` | Path to cuPDLPx build directory |
| `ENABLE_GUROBI` | `ON` (default) — enable Gurobi |
| `ENABLE_HIGHS` | `ON` (default) — enable HiGHS |
| `ENABLE_CUPDLPX` | `ON` (default) — enable GPU LP solver |

### 3. Build

```bash
cmake -S cutLPK/src -B cutLPK/build
cmake --build cutLPK/build --clean-first
```

The binary will be located at `cutLPK/build/cutLPK` (Linux) or `cutLPK/build/Release/cutLPK.exe` (Windows MSVC).

### 4. Runtime library path

Before running, ensure the directories containing the shared libraries of your dependencies are on the search path:

**Linux:**
```bash
export LD_LIBRARY_PATH=/path/to/cuPDLPx/build:/path/to/gurobi/lib:/path/to/highs/lib:$LD_LIBRARY_PATH
```

**Windows (PowerShell):**
```powershell
$env:PATH = "C:\path\to\cuPDLPx\build\Release;C:\path\to\gurobi\bin;C:\path\to\highs\bin;$env:PATH"
```

> Adjust the paths above to match your actual Gurobi version (e.g., `gurobi1103`, `gurobi1300`) and build directories.  macOS is not supported as cuPDLPx requires an NVIDIA GPU.

---

## Command-Line Usage

```
cutLPK <data_file> <K> [key=value ...]
```

The first two arguments are **positional** and **required**: the path to the data CSV (or Laplacian CSV for spectral mode) and the number of clusters $K$. All other options are passed as `key=value` pairs.

### Quick examples

```bash
# Ordinary K-means
./cutLPK iris.csv 3

# Fair K-means (alpha-fair, ρ = 0.8)
./cutLPK HC_data.csv 3 fairness_type=alpha fairness_param=0.8 group_file=HC_labels.csv

# Spectral clustering (reads Laplacian CSV)
./cutLPK football_L.csv 3 is_spectral_clustering=true

# Heuristic only (fair Lloyd, no cutting plane)
./cutLPK Credit_data.csv 4 fairness_type=tau fairness_param=0.25 \
    group_file=Credit_labels.csv heuristic_only=true lloyd_random_starts=1
```

### Complete parameter reference

Parameters marked with ★ are important for reproducing paper experiments.

| Key | Type | Default | Description |
|---|---|---|---|
| **Problem type** | | | |
| `fairness_type` | string | `""` | Fairness constraint: `"alpha"` or `"tau"` |
| `fairness_param` ★ | double | `1.0` | Fairness strength $\rho$. For $\alpha$-fair: set to $\rho$ directly. For $\tau$-fair: set to $\rho/K$ (the C++ code uses the value as-is; $\rho/K$ is computed by the caller) |
| `group_file` ★ | string | `""` | Path to group-labels CSV (one integer group ID per line) |
| `is_spectral_clustering` | bool | `false` | Treat input as Laplacian matrix CSV |
| `heuristic_only` | bool | `false` | Run heuristic only, skip cutting plane |
| **Solver** | | | |
| `solver` | string | `"cupdlpx"` | LP solver backend (`cupdlpx` is the only supported option) |
| `fair_assignment_solver` ★ | string | `"highs"` | Fair assignment subproblem solver: `"gurobi"` or `"highs"` (both support $\alpha$-fair and $\tau$-fair) |
| `solver_warm_start` | bool | `true` | Use primal/dual warm-start across cutting-plane iterations (pass previous LP solution to next LP solve) |
| **Cutting-plane – cut management** | | | |
| `max_cuts_init` ★ | int | `1.5e7` | Max inequalities in the first LP ($p_{\text{init}}$). Fair: $10^6$; spectral: $10^7$ |
| `max_cuts_per_iter` | int | `1e8` | Max total cuts allowed in LP at any iteration |
| `max_cuts_added_iter` | int | `1e7` | Max cuts added in a single iteration |
| `max_separation_size` | int | `1.5e7` | Max number of cuts examined during separation |
| `max_active_cuts_size` | int | `1e8` | Max number of active (tight) cuts retained |
| **Cutting-plane – algorithm control** | | | |
| `max_iter` | int | `3000` | Maximum cutting-plane iterations |
| `num_iter_no_improve` ★ | int | `2` | Stop if gap doesn't improve for this many consecutive iters (fair: `5`) |
| `exact_separation` | bool | `true` | Use greedy separation; `false` for faster heuristic |
| `remove_inactive_cuts` | bool | `true` | Drop cuts not tight at current LP solution |
| `warm_start` | int | `1` | Warm-start cutting plane with the heuristic (Lloyd) solution: `0` = off, `1` = on |
| `t_upper_bound` | int | `K` | Maximum subset size $t$ for inequalities (3k) |
| **Cutting-plane – time limits (seconds)** | | | |
| `initial_lp_time_limit` | double | `360.0` | Time limit for the very first LP |
| `time_limit_lp` | double | `180.0` | Time limit for each subsequent LP |
| `time_limit_all` ★ | double | `7200.0` | Overall wall-clock limit. Paper: **10800** (3 hours) |
| `max_separation_time` | double | `300.0` | Max time in one separation call |
| **Cutting-plane – tolerances** | | | |
| `initial_solver_tol` | double | `1e-6` | Solver tolerance for the first LP |
| `solver_tolerance_per_iter` | double | `1e-6` | Solver tolerance for subsequent LPs |
| `lb_solver_tol` | double | `1e-6` | Tolerance for safe lower-bound computation |
| `cuts_vio_tol` | double | `1e-4` | Violation tolerance for cut separation ($\epsilon_{\text{vio}}$) |
| `cuts_act_tol` | double | `1e-4` | Activity tolerance for tight-cut detection |
| `opt_gap` ★ | double | `1e-4` | Target relative optimality gap ($\epsilon_{\text{opt}}$) |
| **Heuristic** | | | |
| `random_seed` | int | `42` | Random seed |
| `lloyd_random_starts` ★ | int | `100` | Number of random restarts for Lloyd's algorithm |
| **Output** | | | |
| `output_file` | string | auto | Output file stem (auto-generated from data/K if empty) |
| `cutting_plane_verbose` | int | `1` | Console verbosity: `0` = silent, `1` = progress, `2` = debug |
| `output_level` | int | `3` | File output detail: `0` = none, `1` = summary, `2` = iterations, `3` = full |
| **Branch & bound (experimental)** | | | |
| `bnb_node_limit` | int | `0` | Max BnB nodes; `≤0` disables BnB |
| `bnb_time_limit` | double | `3600.0` | BnB time limit (seconds) |
| `bnb_gap_tol` | double | `1e-4` | BnB optimality gap tolerance |
| `bnb_verbose` | int | `1` | BnB verbosity |

---

## Default parameter overrides by problem type

The C++ code applies the following automatic overrides when a problem type is detected:

| Scenario | Override | Reason |
|---|---|---|
| `fairness_type` is set | `max_cuts_init` → $10^6$ | Fewer initial cuts; fairness constraints already tighten the LP |
| `fairness_type` is set | `num_iter_no_improve` → $5$ | Allow more stalled iterations before stopping |
| `is_spectral_clustering=true` | `num_iter_no_improve` → $10^6$ | Effectively disabled; spectral LPs converge differently |

> **CLI arguments always take precedence** over these defaults. If you pass `max_cuts_init=5000000` on the command line, it will NOT be overridden.

---

## Reproducing paper experiments

All experiment scripts live in the `experiments/` directory. Each is a self-contained bash script.  Run with `--dry-run` first to verify paths and job counts.

### Fair K-means clustering (§4.2, Tables 1–3) — 286 instances

```bash
cd experiments
chmod +x reproduce_fair_clustering.sh

# Dry-run
./reproduce_fair_clustering.sh -d /path/to/FairClusteringData --dry-run

# Run all
./reproduce_fair_clustering.sh -d /path/to/FairClusteringData -o ./results_fair

# Resume from job 150
./reproduce_fair_clustering.sh -d /path/to/FairClusteringData -s 150
```

**Datasets:** 10 UCI real-world datasets (HH, HC, Student Math, WDBC, Student Portuguese, Titanic 2, Titanic 3, Credit, Adult2000, Adult3000).  
**Configuration:** $K \in \{2,3,4,5\}$, $\rho \in \{0.99,0.9,0.8,0.7\}$, both $\alpha$-fair and $\tau$-fair constraints.  
**Key parameters:** `max_cuts_init=1000000`, `opt_gap=1e-4`, `time_limit_all=10800`, `fair_assignment_solver=gurobi`.

### Fair Lloyd timing (§4.2, Table fairLloydTiming) — 240 runs

```bash
./reproduce_fair_lloyd_timing.sh -d /path/to/FairClusteringData -o ./timing_results
```

**Datasets:** Credit, Adult2000, Adult3000 (the three largest).  
**Configuration:** $K \in \{2,3,4,5\}$, $\rho \in \{0.99,0.90\}$, both fairness types, 5 seeds each.  
**Mode:** `heuristic_only=true` — Fair Lloyd without cutting plane. Produces `fair_lloyd_raw.csv` and `fair_lloyd_summary.csv` (mean ± std).

### Spectral clustering / community detection (§5.2, Table social_network) — 142 instances

```bash
./reproduce_spectral_clustering.sh -d /path/to/LaplacianCSVs -o ./results_spectral
```

**Datasets:** 16 social/citation networks (polbooks, football, adjnoun, facebook, friendship, deezer_ego, netscience, plus ca-GrQc / ca-HepTh / lastfm_asia at sizes 500/1000/1500).  
**Configuration:** $K \in \{2,\dots,10\}$ (friendship: $K \in \{4,\dots,10\}$).  
**Key parameters:** `max_cuts_init=10000000`, `opt_gap=1e-4`, `time_limit_all=10800`.  

> **Required files:** All 16 Laplacian-matrix CSV files must be placed in a single directory. Run `--help` for the exact filename list.

---

## Dataset sources

All preprocessed datasets are available in a single public Google Drive folder:

[**cutLPK datasets**](https://drive.google.com/drive/folders/1L0CKU7X7z0YZzhKsmlSD5QOy2-b1gXRl?usp=drive_link)

This includes all fair clustering CSV files (data + labels) and all spectral clustering Laplacian CSV files used in the paper experiments.

---

## License

This project is licensed under the MIT License — see [LICENSE.txt](LICENSE.txt).

---

## Citation

If you use cutLPK in your research, please cite:

```bibtex
@misc{khajavirad2026scalablelinearprogrammingbasedframework,
      title={A scalable linear programming-based framework for data clustering}, 
      author={Aida Khajavirad and Huanwen Shen and Yakun Wang},
      year={2026},
      eprint={2607.06709},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2607.06709}, 
}
```
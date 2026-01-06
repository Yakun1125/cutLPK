<p align="center">
  <img src="cutLPK.png" alt="cutLPK Logo" width="300"/>
</p>
A scalable linear programming framework for solving K-Means, Fair K-Means, and Spectral Clustering problems. This project implements a cutting-plane algorithm that utilizing and LP relaxation to solving clustering problems globally, as detailed in [arXiv preprint coming soon].

## Precompiled Binaries

A precompiled cutLPK binary is available for download (tested on Colab only): [cutLPK](https://drive.google.com/file/d/1975-suDKaF1YeWBz9FxslDpohVH5pT_k/view?usp=drive_link).

After downloading, make the binary executable and run it as shown in the examples below.

## Dependencies

This project relies on the following libraries. Versions used for testing are listed for reproducibility.

- **C++ Compiler**: A modern compiler supporting C++17.
- **CMake**: Version 3.16 or newer.
- **cuPDLPx**: A GPU accelerated first order method LP solver. (Tested with CUDA 12.3) [cuPDLPx GitHub Repository](https://github.com/MIT-Lu-Lab/cuPDLPx)
- **Gurobi**: Version 11.0 or newer. A Gurobi license is required. Free academic licenses are available.
- **Eigen**: Version 3.4 or newer. A C++ template library for linear algebra.
- **OpenMP**: For multi-threaded parallelism.

## Building the Project

The project uses CMake for configuration and building. Below are the steps to build the project:
1. Navigate to the project directory:
   ```bash
   cd path/to/cutLPK
   ```
2. Open the `config.cmake` file in a text editor and specify the dependency paths according to your system setup.
3. Create and enter a build directory:
   ```bash
   mkdir build
   cd build
   ```

4. Configure the project with CMake:
   ```bash
   cmake ..
   ```

5. Build the project:
   ```bash
   cmake --build
   ```

## Datasets

- [Social Network datasets](https://drive.google.com/drive/folders/1hL84yrvbdIG-x0TNee8c6dXZ-ZhnCfxy?usp=drive_link)
- [Fair Clustering datasets](https://drive.google.com/drive/folders/1rd5PWDNXlU6rFomKLt509iYVsjLc6FCQ?usp=drive_link)

## Example Usage

* **Ordinary K-Means Clustering**: Provide the data file and specify the number of clusters:

  ```bash
  ./cutLPK iris.csv 3
  ```

* **Fair K-Means Clustering**: Specify the type of fairness, the fairness parameter, and the group label file:

  ```bash
  ./cutLPK HC_data.csv 3 fairness_type=alpha fairness_param=0.8 group_file=HC_labels.csv
  ```

* **Spectral Clustering**: Provide a graph Laplacian file and indicate that the problem is spectral clustering:

  ```bash
  ./cutLPK football_L.csv 3 is_spectral_clustering=true
  ```

## Command-Line Parameters

All solver options are passed as `key=value` pairs after `data_file` and `K`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `solver` | string | `"cupdlpx"` | LP solver. |
| `random_seed` | int | `42` | Seed for randomized components. |
| `fairness_type` | string | empty | Fairness type (`alpha` or `tau`). |
| `fairness_param` | double | `1.0` | Fair clustering strength parameter. |
| `group_file` | string | empty | File with group labels for fair k-means (`fair_clustering_group_file`). |
| `is_spectral_clustering` | bool | `false` | Treat input as Laplacian and run spectral mode (`is_spectral_clustering`). |
| `max_cuts_init` | int | `1.5e7` | Max number of cuts in the first LP. |
| `max_cuts_added_iter` | int | `1e7` | Max cuts added per iteration. |
| `max_separation_size` | int | `1.5e7` | Max separation problem size. |
| `cutting_plane_verbose` | int/bool | `1` | Cutting-plane verbosity. |
| `initial_lp_time_limit` | double | `360.0` | Time limit for the first LP. |
| `time_limit_lp` | double | `180.0` | Time limit (seconds) for each LP. |
| `time_limit_all` | double | `7200.0` | Global time limit (seconds) (`cutting_plane_time_limit`). |
| `solver_tolerance_per_iter` | double | `1e-6` | Solver tolerance per iteration. |
| `cuts_vio_tol` | double | `1e-4` | Cut violation tolerance. |
| `cuts_act_tol` | double | `1e-4` | Cut activation tolerance. |
| `opt_gap` | double | `1e-4` | Target optimality gap. |
| `num_iter_no_improve` | int | `2` | Iterations without improvement before stopping (`cutting_plane_num_iter_no_improve`). |
| `lloyd_random_starts` | int | `100` | Number of random Lloyd starts. |
| `bnb_node_limit` | int | `0` | Max BnB nodes (`<=0` disables BnB) (`bnb_node_limit`). |
| `bnb_time_limit` | double | `3600.0` | BnB time limit (seconds) (`bnb_time_limit`). |
| `bnb_gap_tol` | double | `1e-4` | Relative gap tolerance for BnB (`bnb_gap_tol`). |
| `bnb_verbose` | int | `1` | BnB verbosity level (`bnb_verbose`). |
| `output_file` | string | auto | File name for cutting plane log (defaults to data/K-based name). |
| `bnb_output_file` | string | auto | File name for Branch-and-Bound log (derived from data/K). |
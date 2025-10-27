<p align="center">
  <img src="cutLPK.png" alt="cutLPK Logo" width="300"/>
</p>
A scalable linear programming framework for solving K-Means, Fair K-Means, and Spectral Clustering problems. This project implements a cutting-plane algorithm that utilizing and LP relaxation to solving clustering problems globally, as detailed in [arXiv preprint coming soon].

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

## Example Usage

All commands should be run from the `build` directory.

* **Ordinary K-Means Clustering**: Provide the data file and specify the number of clusters:

  ```bash
  ./cutLPK iris.csv 3
  ```

* **Fair K-Means Clustering**: Specify the type of fairness, the fairness parameter, and the group label file:

  ```bash
  ./cutLPK iris.csv 3 fairness_type=alpha fairness_param=0.8 group_file=groupinfo.csv
  ```

* **Spectral Clustering**: Provide a graph Laplacian file and indicate that the problem is spectral clustering:

  ```bash
  ./cutLPK graph_laplacian.csv 3 is_spectral_clustering=true
  ```

## Key Parameters

The following key parameters can be configured using the format `param_name=param_value`:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| solver | "cupdlp" | The LP solver to use ("cupdlp" or "gurobi") |
| random_seed | 42 | Seed for random number generation |
| max_cuts_init | 1.5e7 | Maximum number of initial cuts |
| max_cuts_added_iter | 3e7 | Maximum number of violated cuts per iteration |
| time_limit_lp | 180 | Time limit for LP in each cutting plane iteration (seconds) |
| time_limit_all | 7200 | Overall time limit (seconds) |
| cuts_vio_tol | 1e-4 | Cut violation tolerance |
| bnb_node_limit | - | Maximum number of nodes to explore in Branch and Bound |
| bnb_time_limit | - | Time limit for Branch and Bound (seconds) |
| bnb_gap_tol | 1e-4 | Gap tolerance for Branch and Bound convergence |
| bnb_verbose | 0 | Branch and Bound verbosity level |
| bnb_output_level | 1 | Detail level of Branch and Bound output |

# cutLPK
A scalable linear programming based algorithm for K-Means clustering

## Dependencies

The project relies on the following libraries:

- **cuPDLP**: A high-performance CUDA-based LP solver. [cuPDLP GitHub Repository](https://github.com/COPT-Public/cuPDLP-C)
- **Eigen**: A C++ template library for linear algebra.
- **OpenMP**: An API for parallel programming in C, C++, and Fortran.

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
   ```bash
   ./cutLPK iris.csv 3
   ```
## Parameters

The following parameters can be set for the algorithm. To change a parameter value, use the format `-param_name=param_value`.

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| solver | "cupdlp" | The LP solver to use |
| output_file | "logFile.txt" | Output file for logs |
| output_level | 1 | Level of output detail |
| random_seed | 1 | Seed for random number generation |
| max_cuts_init | 1.5e7 | Maximum number of initial cuts |
| max_cuts_added_iter | 3e7 | Maximum number of violated cuts per iteration |
| max_separation_size | 1.5e7 | Maximum separation size |
| warm_start | 2 | Warm start option(1 is deterministic choose the initial cuts and 2 is randomly sampling) |
| t_upper_bound | K | Upper bound for t (K is the number of clusters) |
| time_limit_lp | 180 | Time limit for LP in each cutting plane iteration (seconds) |
| time_limit_all | 14400 | Overall time limit (seconds) |
| solver_tolerance_per_iter | 1e-4 | Solver tolerance per iteration |
| cuts_vio_tol | 1e-4 | Cut violation tolerance |
| cuts_act_tol | 1e-4 | Cut activation tolerance |
| opt_gap | 1e-4 | Optimality gap |


## Acknowledgments

This project includes a copy of the cuPDLP-C solver. We have made slight modifications to the termination check to accommodate early termination when solving linear programming problems within our algorithm framework. We recommend using the version of cuPDLP-C included in this project for optimal performance of the algorithm.

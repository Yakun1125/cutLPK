# cutLPK
A scalable linear programming based algorithm for K-Means clustering

## Dependencies

The project relies on the following libraries:

- **cuPDLP**: A high-performance CUDA-based LP solver. [cuPDLP GitHub Repository](https://github.com/COPT-Public/cuPDLP-C)
- **Google OR-Tools PDLP**: A linear solver provided by Google. [Google OR-Tools GitHub Repository](https://github.com/google/or-tools)
- **Gurobi**: A commercial optimization solver.
- **Eigen**: A C++ template library for linear algebra.
- **OpenMP**: An API for parallel programming in C, C++, and Fortran.

## Building the Project

The project uses CMake for configuration and building. Below are the steps to build the project:

   ```bash
   mkdir build
   cd build
   cmake -DBUILD_CUDA=ON ..
   cmake --build . --target plc

## Example Usage

./bin/plc -f iris.csv -c 3

## Acknowledgments

This project includes a copy of the cuPDLP-C solver. We have made slight modifications to the termination check to accommodate early termination when solving linear programming problems within our algorithm framework. As a result, the cuPDLP-C solver is compiled together with the project to ensure these changes are incorporated.

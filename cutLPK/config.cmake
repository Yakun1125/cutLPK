# Options
option(USE_CUPDLP "Use cuPDLP-C" ON)
option(USE_PDLP "Use PDLP" OFF)
option(USE_GUROBI "Use Gurobi" OFF)

# Paths
set(EIGEN_INCLUDE_DIR "" CACHE PATH "Path to Eigen include directory")
set(CMAKE_CUDA_PATH "/usr/local/cuda" CACHE PATH "Path to CUDA installation")
set(HIGHS_HOME "" CACHE PATH "Path to HiGHS installation")
set(CUPDLP_ROOT "" CACHE PATH "Path to cuPDLP-C installation")
set(CUPDLP_LIB_DIR "${CUPDLP_ROOT}/lib")
set(ORTOOLS_ROOT "" CACHE PATH "Path to OR-Tools installation")
set(GUROBI_ROOT "" CACHE PATH "Path to Gurobi installation")

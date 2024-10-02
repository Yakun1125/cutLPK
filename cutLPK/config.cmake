# Options
option(USE_CUPDLP "Use cuPDLP-C" ON)
option(USE_PDLP "Use PDLP" OFF)
option(USE_GUROBI "Use Gurobi" ON)

# Paths
set(EIGEN_INCLUDE_DIR "/content/eigen-3.4.0" CACHE PATH "Path to Eigen include directory")
set(CMAKE_CUDA_PATH "/usr/local/cuda" CACHE PATH "Path to CUDA installation")
set(HIGHS_HOME "/content/HiGHS" CACHE PATH "Path to HiGHS installation")
set(CUPDLP_ROOT "/content/kmeans_cupdlp" CACHE PATH "Path to cuPDLP-C installation")
set(ORTOOLS_ROOT "" CACHE PATH "Path to OR-Tools installation")
set(GUROBI_ROOT "/content/gurobi1102/linux64" CACHE PATH "Path to Gurobi installation")

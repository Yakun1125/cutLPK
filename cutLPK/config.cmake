# Options
option(USE_CUPDLP "Use cuPDLP-C" ON)
option(USE_PDLP "Use PDLP" OFF)
option(USE_GUROBI "Use Gurobi" OFF)

# Paths
set(EIGEN_INCLUDE_DIR "/usr/include/eigen3" CACHE PATH "Path to Eigen include directory")
set(CMAKE_CUDA_PATH "/usr/local/cuda" CACHE PATH "Path to CUDA installation")
set(HIGHS_HOME "/home/yakun/HiGHS" CACHE PATH "Path to HiGHS installation")
set(CUPDLP_ROOT "/home/yakun/.vs/kmeans_cupdlp" CACHE PATH "Path to cuPDLP-C installation")
set(ORTOOLS_ROOT "/home/yakun/or-tools/v9.10" CACHE PATH "Path to OR-Tools installation")
set(GUROBI_ROOT "/home/yakun/GUROBI/gurobi1102/linux64" CACHE PATH "Path to Gurobi installation")
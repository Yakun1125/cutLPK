cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cutLPK) # LANGUAGES CXX CUDA

include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)

set(CUTLPK_CORE_SOURCES
    construct_LPK.cpp
    Lloyd.cpp
    fair_Lloyd.cpp
    Solver_cupdlp.cpp
    separation.cpp
    Rounding_heuristic.cpp
    iterative_cutting_plane.cpp
    spectral_heuristic.cpp
    branch_and_bound.cpp
    ordinary_kmeans_solver.cpp
    fair_kmeans_solver.cpp
)

add_library(cutlpk_core STATIC ${CUTLPK_CORE_SOURCES})
set_target_properties(cutlpk_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(cutlpk_core PUBLIC cxx_std_17)
target_include_directories(cutlpk_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${EIGEN_INCLUDE_DIR}
)

if(NOT DEFINED CUTLPK_BUILD_EXECUTABLE)
    set(CUTLPK_BUILD_EXECUTABLE ON)
endif()

if(CUTLPK_BUILD_EXECUTABLE)
    add_executable(cutLPK main.cpp)
    target_link_libraries(cutLPK PRIVATE cutlpk_core)
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cutlpk_core PUBLIC OpenMP::OpenMP_CXX)
    message(NOTICE "OpenMP enabled")
else()
    message(WARNING "OpenMP not found")
endif()

######################
# Cuda
######################
# CUDA setup
message(NOTICE "Finding CUDA environment")
message(NOTICE "    - CUDA Home detected at ${CMAKE_CUDA_PATH}")
set(CMAKE_CUDA_ARCHITECTURES "all")
set(CMAKE_CUDA_COMPILER "${CMAKE_CUDA_PATH}/bin/nvcc")


# Check if CUDA compiler was found
if(CMAKE_CUDA_COMPILER)
    message(NOTICE "CUDA found: compiler at ${CMAKE_CUDA_COMPILER}")
    target_compile_definitions(cutlpk_core PUBLIC CUDA_AVAILABLE)
else()
  message(FATAL_ERROR "CUDA not found. This project requires CUDA.")
endif()

# Find and link CUDA runtime library
find_library(CUDA_RUNTIME_LIBRARY cudart HINTS ${CMAKE_CUDA_PATH}/lib64)
if(NOT CUDA_RUNTIME_LIBRARY)
  message(FATAL_ERROR "CUDA runtime library not found")
endif()

# Find and link CUDA Sparse library
find_library(CUDA_SPARSE_LIBRARY cusparse HINTS ${CMAKE_CUDA_PATH}/lib64)
if(NOT CUDA_SPARSE_LIBRARY)
  message(FATAL_ERROR "cuSPARSE library not found")
endif()

# Find and link cuBLAS library
find_library(CUDA_BLAS_LIBRARY cublas HINTS ${CMAKE_CUDA_PATH}/lib64)
if(NOT CUDA_BLAS_LIBRARY)
  message(FATAL_ERROR "cuBLAS library not found")
endif()

find_library(CUDA_DRIVER_LIBRARY cuda
        HINTS
                ${CMAKE_CUDA_PATH}/lib64
                ${CMAKE_CUDA_PATH}/lib64/stubs
                /usr/lib/x86_64-linux-gnu)
if(NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR "CUDA driver library not found")
endif()

find_library(CUDA_DEVICE_RUNTIME_LIBRARY cudadevrt HINTS ${CMAKE_CUDA_PATH}/lib64)
if(NOT CUDA_DEVICE_RUNTIME_LIBRARY)
    message(FATAL_ERROR "CUDA device runtime library not found")
endif()

target_include_directories(cutlpk_core PUBLIC ${CMAKE_CUDA_PATH}/include)

######################
# cuPDLPx
######################
set(CUPDLPX_INCLUDE_DIR "${CUPDLPX_ROOT}")
set(CUPDLPX_LIB_DIR "${CUPDLPX_ROOT}/build")

if(ENABLE_CUPDLPX)
    message(STATUS "cuPDLPx support enabled")
    target_include_directories(cutlpk_core PUBLIC ${CUPDLPX_INCLUDE_DIR})
    target_link_directories(cutlpk_core PUBLIC ${CUPDLPX_LIB_DIR})

    find_library(CUPDLPX_LIB cupdlpx PATHS ${CUPDLPX_LIB_DIR} NO_DEFAULT_PATH)

    if(CUPDLPX_LIB)
        target_link_libraries(cutlpk_core PUBLIC ${CUPDLPX_LIB})
        target_compile_definitions(cutlpk_core PUBLIC ENABLE_CUPDLPX)
        message(NOTICE "cuPDLPx library found and linked: ${CUPDLPX_LIB}")
    else()
        message(WARNING "cuPDLPx library not found in ${CUPDLPX_LIB_DIR}")
    endif()
    find_library(ZLIB_LIBRARY z)
    # it's necessary for cupdlpx, but we won't use it actually
    if(ZLIB_LIBRARY)
        target_link_libraries(cutlpk_core PUBLIC ${ZLIB_LIBRARY})
    else()
        message(FATAL_ERROR "zlib not found")
    endif()
else()
    message(STATUS "cuPDLPx support disabled")
endif()

target_link_libraries(cutlpk_core PUBLIC
    ${CUDA_RUNTIME_LIBRARY}
    ${CUDA_SPARSE_LIBRARY}
    ${CUDA_BLAS_LIBRARY}
    ${CUDA_DRIVER_LIBRARY}
    ${CUDA_DEVICE_RUNTIME_LIBRARY})

######################
# Gurobi
######################
target_include_directories(cutlpk_core PUBLIC ${GUROBI_ROOT}/include)
target_link_directories(cutlpk_core PUBLIC ${GUROBI_ROOT}/lib)
target_link_libraries(cutlpk_core PUBLIC 
    ${GUROBI_ROOT}/lib/libgurobi_c++.a
    ${GUROBI_ROOT}/lib/libgurobi130.so
    )
message(NOTICE, " Solver Gurobi enabled ")
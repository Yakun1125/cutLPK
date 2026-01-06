cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cutLPK)

include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)

set(CUTLPK_CORE_SOURCES
    construct_LPK.cpp
    Lloyd.cpp
    fair_Lloyd.cpp
    LP_solvers.cpp
    separation.cpp
    Rounding_heuristic.cpp
    iterative_cutting_plane.cpp
    spectral_heuristic.cpp
    branch_and_bound.cpp
    cutLPK.cpp
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
message(NOTICE "Finding CUDA environment")
message(NOTICE "    - CUDA Home detected at ${CMAKE_CUDA_PATH}")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.23")
        set(CMAKE_CUDA_ARCHITECTURES all)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 89 90 90-virtual)
    endif()
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CUTLPK_CUDA_LIBS
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
)

target_link_libraries(cutlpk_core PUBLIC ${CUTLPK_CUDA_LIBS})

######################
# cuPDLPx
######################
set(CUPDLPX_INCLUDE_DIR "${CUPDLPX_ROOT}/include")
set(CUPDLPX_LIB_DIR "${CUPDLPX_ROOT}/build")
set(PSLP_INTERNAL_INCLUDE "${CUPDLPX_ROOT}/build/_deps/pslp-src/include/PSLP")

if(ENABLE_CUPDLPX)
    message(STATUS "cuPDLPx support enabled")
    target_include_directories(cutlpk_core PUBLIC 
            ${PSLP_INTERNAL_INCLUDE}      
            ${CUPDLPX_ROOT}/include        
            ${CUPDLPX_ROOT}/internal   
        )
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

######################
# Gurobi
######################
target_include_directories(cutlpk_core PUBLIC ${GUROBI_ROOT}/include)
target_link_directories(cutlpk_core PUBLIC ${GUROBI_ROOT}/lib)
target_link_libraries(cutlpk_core PUBLIC 
    ${GUROBI_ROOT}/lib/libgurobi_c++.a
    ${GUROBI_ROOT}/lib/libgurobi130.so
    )
message(NOTICE, " Solver Gurobi for assignment problem enabled ")
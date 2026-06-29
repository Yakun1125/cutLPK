cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cutLPK LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CUDA_PATH)
        set(CUDAToolkit_ROOT "${CMAKE_CUDA_PATH}" CACHE PATH "CUDA toolkit root" FORCE)
endif()

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

# Conditionally add solver-specific sources
if(ENABLE_GUROBI)
    list(APPEND CUTLPK_CORE_SOURCES fair_assignment_gurobi.cpp)
endif()
if(ENABLE_HIGHS)
    list(APPEND CUTLPK_CORE_SOURCES fair_assignment_highs.cpp)
endif()

add_library(cutlpk_core STATIC ${CUTLPK_CORE_SOURCES})
set_target_properties(cutlpk_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(cutlpk_core PUBLIC cxx_std_17)
target_compile_options(cutlpk_core PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /EHsc /utf-8>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native>
)
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
    if(MSVC)
        target_link_options(cutLPK PRIVATE /STACK:67108864)
    endif()
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(cutlpk_core PUBLIC OpenMP::OpenMP_CXX)
message(STATUS "OpenMP enabled")

######################
# Cuda
######################
message(STATUS "Finding CUDA environment")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
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
if(NOT CUPDLPX_BUILD_DIR)
    set(CUPDLPX_BUILD_DIR "${CUPDLPX_ROOT}/build")
endif()
set(PSLP_INTERNAL_INCLUDE "${CUPDLPX_BUILD_DIR}/_deps/pslp-src/include/PSLP")

if(ENABLE_CUPDLPX)
    message(STATUS "cuPDLPx support enabled")
    if(NOT CUPDLPX_ROOT)
        message(FATAL_ERROR "CUPDLPX_ROOT is required when ENABLE_CUPDLPX=ON")
    endif()
    target_include_directories(cutlpk_core PUBLIC 
            ${PSLP_INTERNAL_INCLUDE}      
            ${CUPDLPX_INCLUDE_DIR}        
            ${CUPDLPX_ROOT}/internal
        )
    target_link_directories(cutlpk_core PUBLIC ${CUPDLPX_BUILD_DIR})

    find_library(CUPDLPX_LIB NAMES cupdlpx PATHS
        ${CUPDLPX_BUILD_DIR}
        ${CUPDLPX_BUILD_DIR}/Release
        ${CUPDLPX_BUILD_DIR}/RelWithDebInfo
        NO_DEFAULT_PATH
    )

    if(NOT CUPDLPX_LIB)
        message(FATAL_ERROR "cuPDLPx library not found in ${CUPDLPX_BUILD_DIR}")
    endif()

    find_library(ZLIB_LIBRARY NAMES z zlib zlibstatic PATHS
        ${CUPDLPX_BUILD_DIR}/_deps/zlib-build
        ${CUPDLPX_BUILD_DIR}/_deps/zlib-build/Release
        ${CUPDLPX_BUILD_DIR}/_deps/zlib-build/RelWithDebInfo
    )
    if(NOT ZLIB_LIBRARY)
        message(FATAL_ERROR "zlib not found; set ZLIB_LIBRARY or build cuPDLPx first")
    endif()

    target_link_libraries(cutlpk_core PUBLIC ${CUPDLPX_LIB} ${ZLIB_LIBRARY})
    target_compile_definitions(cutlpk_core PUBLIC ENABLE_CUPDLPX)
    message(STATUS "cuPDLPx library found and linked: ${CUPDLPX_LIB}")
else()
    message(STATUS "cuPDLPx support disabled")
endif()

######################
# Gurobi
######################
if(ENABLE_GUROBI)
    if(NOT GUROBI_ROOT)
        message(FATAL_ERROR "GUROBI_ROOT is required when ENABLE_GUROBI=ON")
    endif()

    target_include_directories(cutlpk_core PUBLIC ${GUROBI_ROOT}/include)
    target_link_directories(cutlpk_core PUBLIC ${GUROBI_ROOT}/lib)

    if(WIN32)
        find_library(GUROBI_CXX_LIB NAMES gurobi_c++md2017 gurobi_c++mt2017 gurobi_c++mdd2017 gurobi_c++mtd2017 PATHS ${GUROBI_ROOT}/lib NO_DEFAULT_PATH)
        find_library(GUROBI_C_LIB NAMES gurobi130 gurobi120 gurobi110 gurobi PATHS ${GUROBI_ROOT}/lib NO_DEFAULT_PATH)
    else()
        find_library(GUROBI_CXX_LIB NAMES gurobi_c++ PATHS ${GUROBI_ROOT}/lib NO_DEFAULT_PATH)
        find_library(GUROBI_C_LIB NAMES gurobi130 gurobi120 gurobi110 gurobi PATHS ${GUROBI_ROOT}/lib NO_DEFAULT_PATH)
    endif()

    if(NOT GUROBI_CXX_LIB OR NOT GUROBI_C_LIB)
        message(FATAL_ERROR "Gurobi libraries not found under ${GUROBI_ROOT}/lib")
    endif()

    target_link_libraries(cutlpk_core PUBLIC ${GUROBI_CXX_LIB} ${GUROBI_C_LIB})
    target_compile_definitions(cutlpk_core PUBLIC ENABLE_GUROBI)
    message(STATUS "Gurobi enabled: ${GUROBI_CXX_LIB}; ${GUROBI_C_LIB}")
else()
    message(STATUS "Gurobi support disabled")
endif()

######################
# HiGHS
######################
if(ENABLE_HIGHS)
    # --- Resolve HIGHS_ROOT ---
    if(NOT HIGHS_ROOT AND DEFINED ENV{HIGHS_ROOT})
        set(HIGHS_ROOT "$ENV{HIGHS_ROOT}")
    endif()
    if(NOT HIGHS_ROOT)
        set(HIGHS_ROOT "/usr/local" CACHE PATH "Path to HiGHS installation or source tree")
    endif()

    # --- Include directories (handle both installed and build-tree layouts) ---
    # Installed layout:  ${HIGHS_ROOT}/include/highs/Highs.h
    # Source layout:     ${HIGHS_ROOT}/highs/Highs.h
    # Build dir (gen):   ${HIGHS_ROOT}/build/HConfig.h or ${HIGHS_ROOT}/build-msvc/HConfig.h
    set(_highs_include_candidates
        "${HIGHS_ROOT}/include"
        "${HIGHS_ROOT}/include/highs"
        "${HIGHS_ROOT}/highs"
        "${HIGHS_ROOT}"
    )
    foreach(_dir ${_highs_include_candidates})
        if(EXISTS "${_dir}/Highs.h" OR EXISTS "${_dir}/highs/Highs.h")
            list(APPEND _highs_inc_dirs "${_dir}")
        endif()
    endforeach()

    # Also add build directory for generated HConfig.h
    set(_highs_build_candidates
        "${HIGHS_ROOT}/build"
        "${HIGHS_ROOT}/build-msvc"
        "${HIGHS_ROOT}/build/Release"
        "${HIGHS_ROOT}/build-msvc/Release"
    )
    foreach(_dir ${_highs_build_candidates})
        if(EXISTS "${_dir}/HConfig.h")
            list(APPEND _highs_inc_dirs "${_dir}")
            break()
        endif()
    endforeach()

    if(NOT _highs_inc_dirs)
        message(WARNING "Could not locate Highs.h. Set HIGHS_ROOT to the HiGHS source or install tree.")
    endif()

    target_include_directories(cutlpk_core PUBLIC ${_highs_inc_dirs})

    # --- Library search ---
    set(_highs_lib_candidates
        "${HIGHS_ROOT}/lib"
        "${HIGHS_ROOT}/build/lib"
        "${HIGHS_ROOT}/build/Release/bin"
        "${HIGHS_ROOT}/build-msvc/Release/bin"
        "${HIGHS_ROOT}/build-msvc/Release/lib"
        "${HIGHS_ROOT}/build/lib/Release"
    )

    if(WIN32)
        find_library(HIGHS_LIB NAMES highs libhighs
            PATHS ${_highs_lib_candidates}
            NO_DEFAULT_PATH)
    else()
        find_library(HIGHS_LIB NAMES highs libhighs
            PATHS ${_highs_lib_candidates}
            NO_DEFAULT_PATH)
    endif()

    if(NOT HIGHS_LIB)
        message(FATAL_ERROR "HiGHS library not found. Build HiGHS first or set HIGHS_ROOT.")
    endif()

    target_link_libraries(cutlpk_core PUBLIC ${HIGHS_LIB})
    target_compile_definitions(cutlpk_core PUBLIC ENABLE_HIGHS)
    message(STATUS "HiGHS enabled: ${HIGHS_LIB}")
    message(STATUS "HiGHS include: ${_highs_inc_dirs}")
else()
    message(STATUS "HiGHS support disabled")
endif()
cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
endif()

project(cutLPK LANGUAGES CXX CUDA)

include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)

# Add all source files to the executable
add_executable(cutLPK 
    main.cpp
    Lloyd.cpp
    cutLPK_Utils.cpp
    fairLPK_Utils.cpp
    cutLPK_Main.cpp
)

include_directories(cutLPK PRIVATE ${EIGEN_INCLUDE_DIR})
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cutLPK PRIVATE OpenMP::OpenMP_CXX)
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


# Find CUDA libraries
find_package(CUDA REQUIRED)
enable_language(CUDA)
if(CUDA_FOUND)
    message(NOTICE "CUDA found: version ${CUDA_VERSION_STRING}")
    set(CUDA_LIBRARY ${CUDA_LIBRARIES})
    add_definitions(-DCUDA_AVAILABLE)
else()
    message(FATAL_ERROR "CUDA not found. This project requires CUDA.")
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

target_include_directories(cutLPK PRIVATE ${CMAKE_CUDA_PATH}/include)
target_link_libraries(cutLPK PRIVATE ${CUDA_LIBRARY} ${CUDA_SPARSE_LIBRARY} ${CUDA_BLAS_LIBRARY})
######################
# Highs
######################
set(HiGHS_HEADER_LIB "${HIGHS_HOME}/build/lib")
set(HiGHS_INCLUDE_DIR "${HIGHS_HOME}/src")
include_directories(${HiGHS_INCLUDE_DIR})
link_directories(${HiGHS_HEADER_LIB})
target_link_directories(cutLPK PRIVATE ${HiGHS_HEADER_LIB})
target_link_libraries(cutLPK PRIVATE highs)

######################
# cuPDLP-C
######################
if(USE_CUPDLP)
   include_directories("${CUPDLP_ROOT}/interface")
   include_directories("${CUPDLP_ROOT}/cupdlp")

   set(CUPDLP_LIB_DIR "${CUPDLP_ROOT}/out/build/linux-release-CUDA/lib")
   link_directories(${CUPDLP_LIB_DIR})
   
   find_library(WRAPPER_HIGHS_LIB wrapper_highs PATHS ${CUPDLP_LIB_DIR} NO_DEFAULT_PATH)
   find_library(WRAPPER_LP_LIB wrapper_lp PATHS ${CUPDLP_LIB_DIR} NO_DEFAULT_PATH)
   find_library(CUPDLP_LIB cupdlp PATHS ${CUPDLP_LIB_DIR} NO_DEFAULT_PATH)
   find_library(CUDALIN_LIB cudalin PATHS ${CUPDLP_LIB_DIR} NO_DEFAULT_PATH)
   
   if(WRAPPER_HIGHS_LIB AND WRAPPER_LP_LIB AND CUPDLP_LIB AND CUDALIN_LIB)
       target_link_libraries(cutLPK PRIVATE 
           ${WRAPPER_HIGHS_LIB} 
           ${WRAPPER_LP_LIB} 
           ${CUPDLP_LIB}
           ${CUDALIN_LIB}
       )
       message(NOTICE "cuPDLP libraries found and linked")
   else()
       message(WARNING "Some cuPDLP libraries not found. 
           WRAPPER_HIGHS_LIB: ${WRAPPER_HIGHS_LIB}, 
           WRAPPER_LP_LIB: ${WRAPPER_LP_LIB}, 
           CUPDLP_LIB: ${CUPDLP_LIB}, 
           CUDALIN_LIB: ${CUDALIN_LIB}"
       )
   endif()
   message(NOTICE " LP solver cuPDLP enabled ")
endif()

######################
# PDLP
######################
if(USE_PDLP)
   list(APPEND CMAKE_PREFIX_PATH ${ORTOOLS_ROOT})
   find_package(ortools REQUIRED)
   target_link_libraries(cutLPK PRIVATE ortools::ortools)
   target_include_directories(cutLPK PRIVATE ${ORTOOLS_ROOT}/include)
   message(NOTICE, " LP solver PDLP enabled ")
endif()

######################
# Gurobi
######################
if(USE_GUROBI)
   include_directories(${GUROBI_ROOT}/include)
   link_directories(${GUROBI_ROOT}/lib)
   target_include_directories(cutLPK PRIVATE ${GUROBI_ROOT}/include)
   target_link_directories(cutLPK PRIVATE ${GUROBI_ROOT}/lib)
   target_link_libraries(cutLPK PRIVATE 
    ${GUROBI_ROOT}/lib/libgurobi_c++.a
    ${GUROBI_ROOT}/lib/libgurobi110.so
    )
   message(NOTICE, " LP solver Gurobi enabled ")
endif()



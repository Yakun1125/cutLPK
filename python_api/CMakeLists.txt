cmake_minimum_required(VERSION 3.15)

project(cutlpk_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CUTLPK_BUILD_EXECUTABLE OFF CACHE BOOL "Build the CLI executable" FORCE)

set(CUTLPK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")
add_subdirectory(${CUTLPK_SOURCE_DIR} cutlpk_core_build)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "For faster LTO builds, set LTO_NTHREADS to the number of CPU cores, e.g.: export LTO_NTHREADS=$(nproc)")
endif()

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_cutlpk MODULE bindings.cpp)

target_link_libraries(_cutlpk PRIVATE cutlpk_core)
target_include_directories(_cutlpk PRIVATE ${CUTLPK_SOURCE_DIR})
set_target_properties(_cutlpk PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cutlpk
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cutlpk
)
install(TARGETS _cutlpk LIBRARY DESTINATION cutlpk)
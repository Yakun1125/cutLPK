cmake_minimum_required(VERSION 3.15)

project(cutlpk_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CUTLPK_BUILD_EXECUTABLE OFF CACHE BOOL "Build the CLI executable" FORCE)

# -------------------------------------------------------------------
# PyPI wheel defaults – Gurobi's license prohibits redistribution on
# PyPI, so we default to HiGHS (MIT) for fair assignment.  Users
# building from source can override with -DENABLE_GUROBI=ON.
# -------------------------------------------------------------------
set(ENABLE_GUROBI OFF CACHE BOOL "Enable Gurobi (not redistributable on PyPI)" FORCE)
set(ENABLE_HIGHS  ON  CACHE BOOL "Enable HiGHS (MIT – safe for PyPI)" FORCE)

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

# Bundle cuPDLPx shared library into the wheel
if(WIN32)
    # Windows: find cupdlpx.dll
    file(GLOB CUPDLPX_DLL_FILES "${CUPDLPX_BUILD_DIR}/Release/cupdlpx.dll"
                                 "${CUPDLPX_BUILD_DIR}/cupdlpx.dll"
                                 "${CUPDLPX_BUILD_DIR}/bin/cupdlpx.dll")
    if(CUPDLPX_DLL_FILES)
        list(GET CUPDLPX_DLL_FILES 0 CUPDLPX_DLL)
        install(FILES "${CUPDLPX_DLL}" DESTINATION cutlpk)
        message(STATUS "Bundling cuPDLPx DLL: ${CUPDLPX_DLL}")
    endif()
else()
    # Linux: find libcupdlpx.so
    if(EXISTS "${CUPDLPX_BUILD_DIR}/libcupdlpx.so")
        install(FILES "${CUPDLPX_BUILD_DIR}/libcupdlpx.so" DESTINATION cutlpk)
    elseif(EXISTS "${CUPDLPX_BUILD_DIR}/cupdlpx.so")
        install(FILES "${CUPDLPX_BUILD_DIR}/cupdlpx.so" DESTINATION cutlpk RENAME libcupdlpx.so)
    endif()
endif()

# Bundle PSLP dependency if present
if(WIN32)
    file(GLOB PSLP_DLL_FILES "${CUPDLPX_BUILD_DIR}/Release/pslp.dll"
                              "${CUPDLPX_BUILD_DIR}/_deps/pslp-build/Release/pslp.dll"
                              "${CUPDLPX_BUILD_DIR}/_deps/pslp-build/*/Release/pslp.dll")
    if(PSLP_DLL_FILES)
        list(GET PSLP_DLL_FILES 0 PSLP_DLL)
        install(FILES "${PSLP_DLL}" DESTINATION cutlpk)
        message(STATUS "Bundling PSLP DLL: ${PSLP_DLL}")
    endif()
else()
    file(GLOB PSLP_FILES "${CUPDLPX_BUILD_DIR}/_deps/pslp-build/*/libPSLP.so"
                         "${CUPDLPX_BUILD_DIR}/_deps/pslp-build/libPSLP.so"
                         "${CUPDLPX_BUILD_DIR}/libPSLP.so")
    if(PSLP_FILES)
        list(GET PSLP_FILES 0 PSLP_SO)
        install(FILES "${PSLP_SO}" DESTINATION cutlpk)
        message(STATUS "Bundling PSLP: ${PSLP_SO}")
    endif()
endif()

# Bundle HiGHS shared library (MIT license — freely redistributable)
if(ENABLE_HIGHS)
    if(WIN32)
        # On Windows, HIGHS_LIB points to the .lib import library;
        # we need the .dll alongside it.
        get_filename_component(_highs_dir "${HIGHS_LIB}" DIRECTORY)
        set(_highs_dll "${_highs_dir}/highs.dll")
        if(EXISTS "${_highs_dll}")
            install(FILES "${_highs_dll}" DESTINATION cutlpk)
            message(STATUS "Bundling HiGHS DLL: ${_highs_dll}")
        else()
            message(WARNING "HiGHS DLL not found at ${_highs_dll}")
        endif()
    else()
        # Linux: install the .so directly
        if(HIGHS_LIB AND EXISTS "${HIGHS_LIB}")
            install(FILES "${HIGHS_LIB}" DESTINATION cutlpk)
            message(STATUS "Bundling HiGHS: ${HIGHS_LIB}")
        elseif(HIGHS_ROOT)
            file(GLOB HIGHS_SO_FILES "${HIGHS_ROOT}/build/lib/libhighs.so*"
                                     "${HIGHS_ROOT}/lib/libhighs.so*")
            if(HIGHS_SO_FILES)
                list(GET HIGHS_SO_FILES 0 HIGHS_SO)
                install(FILES "${HIGHS_SO}" DESTINATION cutlpk)
                message(STATUS "Bundling HiGHS: ${HIGHS_SO}")
            endif()
        endif()
    endif()
endif()

# Set rpath on Linux/macOS so _cutlpk finds bundled .so files in its own directory.
# On Windows, DLLs are found via the same-directory rule (no rpath needed).
if(NOT WIN32)
    if(ENABLE_GUROBI)
        set_target_properties(_cutlpk PROPERTIES
            INSTALL_RPATH "$ORIGIN:$ORIGIN/../gurobipy/.libs"
            BUILD_WITH_INSTALL_RPATH TRUE)
    else()
        set_target_properties(_cutlpk PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE)
    endif()
endif()
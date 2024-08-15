enable_language(CXX)

#############################################
# enable LP parsers
#############################################
add_library(wrapper_lp SHARED
        mps_lp.h mps_lp.c
)
target_link_libraries(
        wrapper_lp PUBLIC cupdlp
)
if (${CUDA_LIBRARY-NOTFOUND})
    message(NOTICE "- CPU version PDLP")
    target_compile_definitions(wrapper_lp
            PUBLIC
            -DCUPDLP_CPU
    )
endif()

set(EIGEN_INCLUDE_DIR "/usr/include/eigen3")
list(APPEND CMAKE_PREFIX_PATH "/home/yakun/or-tools/v9.6")
find_package(ortools 9.6 REQUIRED)

# Set the path to the Gurobi installation
set(GUROBI_HOME "/home/yakun/GUROBI/gurobi1102/linux64")

# Include directories for Gurobi
include_directories(${GUROBI_HOME}/include)
# Link directories for Gurobi
link_directories(${GUROBI_HOME}/lib)


#############################################
# enable HiGHS input targets
#############################################

message(NOTICE "--------------------- cuPDLP  HiGHS  CONFIG -----------------------")
include(../FindHiGHSConf.cmake)
add_library(wrapper_highs SHARED
        mps_lp.h mps_lp.c
        wrapper_highs.cpp ${HiGHS_HEADER_FILES}
        wrapper_highs.h)

target_include_directories(
        wrapper_highs PUBLIC
        ${EIGEN_INCLUDE_DIR}
)

target_include_directories(
        wrapper_highs PUBLIC
        ${HiGHS_INCLUDE_DIR}
)
target_link_libraries(
        wrapper_highs PUBLIC cupdlp
        ${HiGHS_LIBRARY}
)
if (${CUDA_LIBRARY-NOTFOUND})
    message(NOTICE "- CPU version PDLP")
    target_compile_definitions(wrapper_highs
            PUBLIC
            -DCUPDLP_CPU
    )
endif()
#############################################
# enable lpk solve
#############################################
add_library(wrapper_lpk SHARED
    wrapper_lpk.cpp
    wrapper_lpk.h
)

target_include_directories(wrapper_lpk PUBLIC
    ${EIGEN_INCLUDE_DIR}
    ${HiGHS_INCLUDE_DIR}
    ${GUROBI_HOME}/include
)
# get openmp
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(wrapper_lpk PUBLIC OpenMP::OpenMP_CXX)
endif()

# Ensure wrapper_lpk is linked against wrapper_highs since it's using it
target_link_libraries(wrapper_lpk PUBLIC
    wrapper_highs
    ortools::ortools
)

target_link_libraries(wrapper_lpk PRIVATE ${GUROBI_HOME}/lib/libgurobi_c++.a ${GUROBI_HOME}/lib/libgurobi110.so)


#############################################
# cuPDLP-C binary using HiGHS as parsers
#############################################
add_executable(plc 
        ${HiGHS_HEADER_FILES}
        lpk_pdlp.c
)
target_include_directories(
        plc PRIVATE
        ${HiGHS_INCLUDE_DIR}
        )
target_include_directories(plc PRIVATE ${EIGEN_INCLUDE_DIR})
target_link_libraries(
        plc PUBLIC
        wrapper_highs
        wrapper_lpk
        ${HiGHS_LIBRARY}
)


if (${CUDA_LIBRARY-NOTFOUND})
    target_link_libraries(plc PRIVATE cupdlp m)
else ()
    target_link_libraries(plc PRIVATE cupdlp ${CUDA_LIBRARY} m)
endif ()
if (${CUDA_LIBRARY-NOTFOUND})
    add_custom_target(
            deploy
            COMMAND mv "${CMAKE_BINARY_DIR}/bin/plc" "${CMAKE_BINARY_DIR}/bin/plccpu"
            DEPENDS plc plc
    )
else()
    add_custom_target(
            deploy
            COMMAND mv "${CMAKE_BINARY_DIR}/bin/plc" "${CMAKE_BINARY_DIR}/bin/plcgpu"
            DEPENDS plc plc
    )
endif ()
# (c) Tianhao Liu
set(HiGHS_LIBRARY-NOTFOUND, OFF)
message(NOTICE "Finding HiGHS environment")
set(HIGHS_HOME "/home/yakun/HiGHS")
message(NOTICE "    - HiGHS Home detected at ${HIGHS_HOME}")
set(CMAKE_HiGHS_PATH "${HIGHS_HOME}")
set(HiGHS_HEADER_LIB "${HIGHS_HOME}/build/lib")
set(HiGHS_INCLUDE_DIR "${HIGHS_HOME}/src")

file(GLOB_RECURSE
        HiGHS_HEADER_FILES
        "${HiGHS_INCLUDE_DIR}/*.h"
        "${HiGHS_INCLUDE_DIR}/*.hpp"
)

find_library(HiGHS_LIBRARY
        NAMES highs
        PATHS "${HiGHS_HEADER_LIB}"
        REQUIRED
        NO_DEFAULT_PATH
)

message(NOTICE
        "    - HiGHS Libraries detected at ${HiGHS_LIBRARY}")
message(NOTICE
        "    - HiGHS include dir at ${HiGHS_INCLUDE_DIR}")

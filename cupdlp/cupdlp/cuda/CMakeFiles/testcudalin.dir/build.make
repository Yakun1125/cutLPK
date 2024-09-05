# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yakun/cuPDLP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yakun/cuPDLP

# Include any dependencies generated for this target.
include cupdlp/cuda/CMakeFiles/testcudalin.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include cupdlp/cuda/CMakeFiles/testcudalin.dir/compiler_depend.make

# Include the progress variables for this target.
include cupdlp/cuda/CMakeFiles/testcudalin.dir/progress.make

# Include the compile flags for this target's objects.
include cupdlp/cuda/CMakeFiles/testcudalin.dir/flags.make

cupdlp/cuda/CMakeFiles/testcudalin.dir/test_cuda_linalg.c.o: cupdlp/cuda/CMakeFiles/testcudalin.dir/flags.make
cupdlp/cuda/CMakeFiles/testcudalin.dir/test_cuda_linalg.c.o: cupdlp/cuda/test_cuda_linalg.c
cupdlp/cuda/CMakeFiles/testcudalin.dir/test_cuda_linalg.c.o: cupdlp/cuda/CMakeFiles/testcudalin.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/yakun/cuPDLP/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object cupdlp/cuda/CMakeFiles/testcudalin.dir/test_cuda_linalg.c.o"
	cd /home/yakun/cuPDLP/cupdlp/cuda && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT cupdlp/cuda/CMakeFiles/testcudalin.dir/test_cuda_linalg.c.o -MF CMakeFiles/testcudalin.dir/test_cuda_linalg.c.o.d -o CMakeFiles/testcudalin.dir/test_cuda_linalg.c.o -c /home/yakun/cuPDLP/cupdlp/cuda/test_cuda_linalg.c

cupdlp/cuda/CMakeFiles/testcudalin.dir/test_cuda_linalg.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/testcudalin.dir/test_cuda_linalg.c.i"
	cd /home/yakun/cuPDLP/cupdlp/cuda && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/yakun/cuPDLP/cupdlp/cuda/test_cuda_linalg.c > CMakeFiles/testcudalin.dir/test_cuda_linalg.c.i

cupdlp/cuda/CMakeFiles/testcudalin.dir/test_cuda_linalg.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/testcudalin.dir/test_cuda_linalg.c.s"
	cd /home/yakun/cuPDLP/cupdlp/cuda && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/yakun/cuPDLP/cupdlp/cuda/test_cuda_linalg.c -o CMakeFiles/testcudalin.dir/test_cuda_linalg.c.s

# Object files for target testcudalin
testcudalin_OBJECTS = \
"CMakeFiles/testcudalin.dir/test_cuda_linalg.c.o"

# External object files for target testcudalin
testcudalin_EXTERNAL_OBJECTS =

bin/testcudalin: cupdlp/cuda/CMakeFiles/testcudalin.dir/test_cuda_linalg.c.o
bin/testcudalin: cupdlp/cuda/CMakeFiles/testcudalin.dir/build.make
bin/testcudalin: lib/libcudalin.so
bin/testcudalin: /usr/local/cuda/lib64/libcudart.so
bin/testcudalin: /usr/local/cuda/lib64/libcusparse.so
bin/testcudalin: /usr/local/cuda/lib64/libcublas.so
bin/testcudalin: cupdlp/cuda/CMakeFiles/testcudalin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/yakun/cuPDLP/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ../../bin/testcudalin"
	cd /home/yakun/cuPDLP/cupdlp/cuda && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testcudalin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cupdlp/cuda/CMakeFiles/testcudalin.dir/build: bin/testcudalin
.PHONY : cupdlp/cuda/CMakeFiles/testcudalin.dir/build

cupdlp/cuda/CMakeFiles/testcudalin.dir/clean:
	cd /home/yakun/cuPDLP/cupdlp/cuda && $(CMAKE_COMMAND) -P CMakeFiles/testcudalin.dir/cmake_clean.cmake
.PHONY : cupdlp/cuda/CMakeFiles/testcudalin.dir/clean

cupdlp/cuda/CMakeFiles/testcudalin.dir/depend:
	cd /home/yakun/cuPDLP && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yakun/cuPDLP /home/yakun/cuPDLP/cupdlp/cuda /home/yakun/cuPDLP /home/yakun/cuPDLP/cupdlp/cuda /home/yakun/cuPDLP/cupdlp/cuda/CMakeFiles/testcudalin.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : cupdlp/cuda/CMakeFiles/testcudalin.dir/depend

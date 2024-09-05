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
include cupdlp/cuda/CMakeFiles/testcublas.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include cupdlp/cuda/CMakeFiles/testcublas.dir/compiler_depend.make

# Include the progress variables for this target.
include cupdlp/cuda/CMakeFiles/testcublas.dir/progress.make

# Include the compile flags for this target's objects.
include cupdlp/cuda/CMakeFiles/testcublas.dir/flags.make

cupdlp/cuda/CMakeFiles/testcublas.dir/test_cublas.c.o: cupdlp/cuda/CMakeFiles/testcublas.dir/flags.make
cupdlp/cuda/CMakeFiles/testcublas.dir/test_cublas.c.o: cupdlp/cuda/test_cublas.c
cupdlp/cuda/CMakeFiles/testcublas.dir/test_cublas.c.o: cupdlp/cuda/CMakeFiles/testcublas.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/yakun/cuPDLP/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object cupdlp/cuda/CMakeFiles/testcublas.dir/test_cublas.c.o"
	cd /home/yakun/cuPDLP/cupdlp/cuda && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT cupdlp/cuda/CMakeFiles/testcublas.dir/test_cublas.c.o -MF CMakeFiles/testcublas.dir/test_cublas.c.o.d -o CMakeFiles/testcublas.dir/test_cublas.c.o -c /home/yakun/cuPDLP/cupdlp/cuda/test_cublas.c

cupdlp/cuda/CMakeFiles/testcublas.dir/test_cublas.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/testcublas.dir/test_cublas.c.i"
	cd /home/yakun/cuPDLP/cupdlp/cuda && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/yakun/cuPDLP/cupdlp/cuda/test_cublas.c > CMakeFiles/testcublas.dir/test_cublas.c.i

cupdlp/cuda/CMakeFiles/testcublas.dir/test_cublas.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/testcublas.dir/test_cublas.c.s"
	cd /home/yakun/cuPDLP/cupdlp/cuda && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/yakun/cuPDLP/cupdlp/cuda/test_cublas.c -o CMakeFiles/testcublas.dir/test_cublas.c.s

# Object files for target testcublas
testcublas_OBJECTS = \
"CMakeFiles/testcublas.dir/test_cublas.c.o"

# External object files for target testcublas
testcublas_EXTERNAL_OBJECTS =

bin/testcublas: cupdlp/cuda/CMakeFiles/testcublas.dir/test_cublas.c.o
bin/testcublas: cupdlp/cuda/CMakeFiles/testcublas.dir/build.make
bin/testcublas: lib/libcudalin.so
bin/testcublas: /usr/local/cuda/lib64/libcudart.so
bin/testcublas: /usr/local/cuda/lib64/libcusparse.so
bin/testcublas: /usr/local/cuda/lib64/libcublas.so
bin/testcublas: cupdlp/cuda/CMakeFiles/testcublas.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/yakun/cuPDLP/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ../../bin/testcublas"
	cd /home/yakun/cuPDLP/cupdlp/cuda && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testcublas.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cupdlp/cuda/CMakeFiles/testcublas.dir/build: bin/testcublas
.PHONY : cupdlp/cuda/CMakeFiles/testcublas.dir/build

cupdlp/cuda/CMakeFiles/testcublas.dir/clean:
	cd /home/yakun/cuPDLP/cupdlp/cuda && $(CMAKE_COMMAND) -P CMakeFiles/testcublas.dir/cmake_clean.cmake
.PHONY : cupdlp/cuda/CMakeFiles/testcublas.dir/clean

cupdlp/cuda/CMakeFiles/testcublas.dir/depend:
	cd /home/yakun/cuPDLP && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yakun/cuPDLP /home/yakun/cuPDLP/cupdlp/cuda /home/yakun/cuPDLP /home/yakun/cuPDLP/cupdlp/cuda /home/yakun/cuPDLP/cupdlp/cuda/CMakeFiles/testcublas.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : cupdlp/cuda/CMakeFiles/testcublas.dir/depend

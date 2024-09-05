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
include cupdlp/cuda/CMakeFiles/cudalin.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include cupdlp/cuda/CMakeFiles/cudalin.dir/compiler_depend.make

# Include the progress variables for this target.
include cupdlp/cuda/CMakeFiles/cudalin.dir/progress.make

# Include the compile flags for this target's objects.
include cupdlp/cuda/CMakeFiles/cudalin.dir/flags.make

cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.o: cupdlp/cuda/CMakeFiles/cudalin.dir/flags.make
cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.o: cupdlp/cuda/CMakeFiles/cudalin.dir/includes_CUDA.rsp
cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.o: cupdlp/cuda/cupdlp_cuda_kernels.cu
cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.o: cupdlp/cuda/CMakeFiles/cudalin.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/yakun/cuPDLP/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.o"
	cd /home/yakun/cuPDLP/cupdlp/cuda && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.o -MF CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.o.d -x cu -rdc=true -c /home/yakun/cuPDLP/cupdlp/cuda/cupdlp_cuda_kernels.cu -o CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.o

cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.o: cupdlp/cuda/CMakeFiles/cudalin.dir/flags.make
cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.o: cupdlp/cuda/CMakeFiles/cudalin.dir/includes_CUDA.rsp
cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.o: cupdlp/cuda/cupdlp_cudalinalg.cu
cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.o: cupdlp/cuda/CMakeFiles/cudalin.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/yakun/cuPDLP/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.o"
	cd /home/yakun/cuPDLP/cupdlp/cuda && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.o -MF CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.o.d -x cu -rdc=true -c /home/yakun/cuPDLP/cupdlp/cuda/cupdlp_cudalinalg.cu -o CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.o

cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cudalin
cudalin_OBJECTS = \
"CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.o" \
"CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.o"

# External object files for target cudalin
cudalin_EXTERNAL_OBJECTS =

cupdlp/cuda/CMakeFiles/cudalin.dir/cmake_device_link.o: cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.o
cupdlp/cuda/CMakeFiles/cudalin.dir/cmake_device_link.o: cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.o
cupdlp/cuda/CMakeFiles/cudalin.dir/cmake_device_link.o: cupdlp/cuda/CMakeFiles/cudalin.dir/build.make
cupdlp/cuda/CMakeFiles/cudalin.dir/cmake_device_link.o: /usr/local/cuda/lib64/libcudart.so
cupdlp/cuda/CMakeFiles/cudalin.dir/cmake_device_link.o: /usr/local/cuda/lib64/libcusparse.so
cupdlp/cuda/CMakeFiles/cudalin.dir/cmake_device_link.o: /usr/local/cuda/lib64/libcublas.so
cupdlp/cuda/CMakeFiles/cudalin.dir/cmake_device_link.o: cupdlp/cuda/CMakeFiles/cudalin.dir/deviceLinkLibs.rsp
cupdlp/cuda/CMakeFiles/cudalin.dir/cmake_device_link.o: cupdlp/cuda/CMakeFiles/cudalin.dir/deviceObjects1.rsp
cupdlp/cuda/CMakeFiles/cudalin.dir/cmake_device_link.o: cupdlp/cuda/CMakeFiles/cudalin.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/yakun/cuPDLP/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/cudalin.dir/cmake_device_link.o"
	cd /home/yakun/cuPDLP/cupdlp/cuda && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cudalin.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cupdlp/cuda/CMakeFiles/cudalin.dir/build: cupdlp/cuda/CMakeFiles/cudalin.dir/cmake_device_link.o
.PHONY : cupdlp/cuda/CMakeFiles/cudalin.dir/build

# Object files for target cudalin
cudalin_OBJECTS = \
"CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.o" \
"CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.o"

# External object files for target cudalin
cudalin_EXTERNAL_OBJECTS =

lib/libcudalin.so: cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cuda_kernels.cu.o
lib/libcudalin.so: cupdlp/cuda/CMakeFiles/cudalin.dir/cupdlp_cudalinalg.cu.o
lib/libcudalin.so: cupdlp/cuda/CMakeFiles/cudalin.dir/build.make
lib/libcudalin.so: /usr/local/cuda/lib64/libcudart.so
lib/libcudalin.so: /usr/local/cuda/lib64/libcusparse.so
lib/libcudalin.so: /usr/local/cuda/lib64/libcublas.so
lib/libcudalin.so: cupdlp/cuda/CMakeFiles/cudalin.dir/cmake_device_link.o
lib/libcudalin.so: cupdlp/cuda/CMakeFiles/cudalin.dir/linkLibs.rsp
lib/libcudalin.so: cupdlp/cuda/CMakeFiles/cudalin.dir/objects1.rsp
lib/libcudalin.so: cupdlp/cuda/CMakeFiles/cudalin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/yakun/cuPDLP/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CUDA shared library ../../lib/libcudalin.so"
	cd /home/yakun/cuPDLP/cupdlp/cuda && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cudalin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cupdlp/cuda/CMakeFiles/cudalin.dir/build: lib/libcudalin.so
.PHONY : cupdlp/cuda/CMakeFiles/cudalin.dir/build

cupdlp/cuda/CMakeFiles/cudalin.dir/clean:
	cd /home/yakun/cuPDLP/cupdlp/cuda && $(CMAKE_COMMAND) -P CMakeFiles/cudalin.dir/cmake_clean.cmake
.PHONY : cupdlp/cuda/CMakeFiles/cudalin.dir/clean

cupdlp/cuda/CMakeFiles/cudalin.dir/depend:
	cd /home/yakun/cuPDLP && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yakun/cuPDLP /home/yakun/cuPDLP/cupdlp/cuda /home/yakun/cuPDLP /home/yakun/cuPDLP/cupdlp/cuda /home/yakun/cuPDLP/cupdlp/cuda/CMakeFiles/cudalin.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : cupdlp/cuda/CMakeFiles/cudalin.dir/depend

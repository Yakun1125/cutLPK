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
include interface/CMakeFiles/wrapper_lp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include interface/CMakeFiles/wrapper_lp.dir/compiler_depend.make

# Include the progress variables for this target.
include interface/CMakeFiles/wrapper_lp.dir/progress.make

# Include the compile flags for this target's objects.
include interface/CMakeFiles/wrapper_lp.dir/flags.make

interface/CMakeFiles/wrapper_lp.dir/mps_lp.c.o: interface/CMakeFiles/wrapper_lp.dir/flags.make
interface/CMakeFiles/wrapper_lp.dir/mps_lp.c.o: interface/mps_lp.c
interface/CMakeFiles/wrapper_lp.dir/mps_lp.c.o: interface/CMakeFiles/wrapper_lp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/yakun/cuPDLP/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object interface/CMakeFiles/wrapper_lp.dir/mps_lp.c.o"
	cd /home/yakun/cuPDLP/interface && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT interface/CMakeFiles/wrapper_lp.dir/mps_lp.c.o -MF CMakeFiles/wrapper_lp.dir/mps_lp.c.o.d -o CMakeFiles/wrapper_lp.dir/mps_lp.c.o -c /home/yakun/cuPDLP/interface/mps_lp.c

interface/CMakeFiles/wrapper_lp.dir/mps_lp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/wrapper_lp.dir/mps_lp.c.i"
	cd /home/yakun/cuPDLP/interface && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/yakun/cuPDLP/interface/mps_lp.c > CMakeFiles/wrapper_lp.dir/mps_lp.c.i

interface/CMakeFiles/wrapper_lp.dir/mps_lp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/wrapper_lp.dir/mps_lp.c.s"
	cd /home/yakun/cuPDLP/interface && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/yakun/cuPDLP/interface/mps_lp.c -o CMakeFiles/wrapper_lp.dir/mps_lp.c.s

# Object files for target wrapper_lp
wrapper_lp_OBJECTS = \
"CMakeFiles/wrapper_lp.dir/mps_lp.c.o"

# External object files for target wrapper_lp
wrapper_lp_EXTERNAL_OBJECTS =

lib/libwrapper_lp.so: interface/CMakeFiles/wrapper_lp.dir/mps_lp.c.o
lib/libwrapper_lp.so: interface/CMakeFiles/wrapper_lp.dir/build.make
lib/libwrapper_lp.so: lib/libcupdlp.so
lib/libwrapper_lp.so: interface/CMakeFiles/wrapper_lp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/yakun/cuPDLP/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C shared library ../lib/libwrapper_lp.so"
	cd /home/yakun/cuPDLP/interface && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/wrapper_lp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
interface/CMakeFiles/wrapper_lp.dir/build: lib/libwrapper_lp.so
.PHONY : interface/CMakeFiles/wrapper_lp.dir/build

interface/CMakeFiles/wrapper_lp.dir/clean:
	cd /home/yakun/cuPDLP/interface && $(CMAKE_COMMAND) -P CMakeFiles/wrapper_lp.dir/cmake_clean.cmake
.PHONY : interface/CMakeFiles/wrapper_lp.dir/clean

interface/CMakeFiles/wrapper_lp.dir/depend:
	cd /home/yakun/cuPDLP && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yakun/cuPDLP /home/yakun/cuPDLP/interface /home/yakun/cuPDLP /home/yakun/cuPDLP/interface /home/yakun/cuPDLP/interface/CMakeFiles/wrapper_lp.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : interface/CMakeFiles/wrapper_lp.dir/depend


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

# Utility rule file for deploy.

# Include any custom commands dependencies for this target.
include interface/CMakeFiles/deploy.dir/compiler_depend.make

# Include the progress variables for this target.
include interface/CMakeFiles/deploy.dir/progress.make

interface/CMakeFiles/deploy: bin/plc
interface/CMakeFiles/deploy: bin/plc
	cd /home/yakun/cuPDLP/interface && mv /home/yakun/cuPDLP/bin/plc /home/yakun/cuPDLP/bin/plcgpu

deploy: interface/CMakeFiles/deploy
deploy: interface/CMakeFiles/deploy.dir/build.make
.PHONY : deploy

# Rule to build all files generated by this target.
interface/CMakeFiles/deploy.dir/build: deploy
.PHONY : interface/CMakeFiles/deploy.dir/build

interface/CMakeFiles/deploy.dir/clean:
	cd /home/yakun/cuPDLP/interface && $(CMAKE_COMMAND) -P CMakeFiles/deploy.dir/cmake_clean.cmake
.PHONY : interface/CMakeFiles/deploy.dir/clean

interface/CMakeFiles/deploy.dir/depend:
	cd /home/yakun/cuPDLP && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yakun/cuPDLP /home/yakun/cuPDLP/interface /home/yakun/cuPDLP /home/yakun/cuPDLP/interface /home/yakun/cuPDLP/interface/CMakeFiles/deploy.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : interface/CMakeFiles/deploy.dir/depend

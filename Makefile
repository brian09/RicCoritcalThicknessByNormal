# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/brian/RicCorticalThicknessByNormal

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/brian/RicCorticalThicknessByNormal

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running interactive CMake command-line interface..."
	/usr/bin/cmake -i .
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/brian/RicCorticalThicknessByNormal/CMakeFiles /home/brian/RicCorticalThicknessByNormal/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/brian/RicCorticalThicknessByNormal/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named RicCorticalThicknessByNormal

# Build rule for target.
RicCorticalThicknessByNormal: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 RicCorticalThicknessByNormal
.PHONY : RicCorticalThicknessByNormal

# fast build rule for target.
RicCorticalThicknessByNormal/fast:
	$(MAKE) -f CMakeFiles/RicCorticalThicknessByNormal.dir/build.make CMakeFiles/RicCorticalThicknessByNormal.dir/build
.PHONY : RicCorticalThicknessByNormal/fast

GM_Normal.o: GM_Normal.cpp.o
.PHONY : GM_Normal.o

# target to build an object file
GM_Normal.cpp.o:
	$(MAKE) -f CMakeFiles/RicCorticalThicknessByNormal.dir/build.make CMakeFiles/RicCorticalThicknessByNormal.dir/GM_Normal.cpp.o
.PHONY : GM_Normal.cpp.o

GM_Normal.i: GM_Normal.cpp.i
.PHONY : GM_Normal.i

# target to preprocess a source file
GM_Normal.cpp.i:
	$(MAKE) -f CMakeFiles/RicCorticalThicknessByNormal.dir/build.make CMakeFiles/RicCorticalThicknessByNormal.dir/GM_Normal.cpp.i
.PHONY : GM_Normal.cpp.i

GM_Normal.s: GM_Normal.cpp.s
.PHONY : GM_Normal.s

# target to generate assembly for a file
GM_Normal.cpp.s:
	$(MAKE) -f CMakeFiles/RicCorticalThicknessByNormal.dir/build.make CMakeFiles/RicCorticalThicknessByNormal.dir/GM_Normal.cpp.s
.PHONY : GM_Normal.cpp.s

RicCorticalThicknessByNormal.o: RicCorticalThicknessByNormal.cpp.o
.PHONY : RicCorticalThicknessByNormal.o

# target to build an object file
RicCorticalThicknessByNormal.cpp.o:
	$(MAKE) -f CMakeFiles/RicCorticalThicknessByNormal.dir/build.make CMakeFiles/RicCorticalThicknessByNormal.dir/RicCorticalThicknessByNormal.cpp.o
.PHONY : RicCorticalThicknessByNormal.cpp.o

RicCorticalThicknessByNormal.i: RicCorticalThicknessByNormal.cpp.i
.PHONY : RicCorticalThicknessByNormal.i

# target to preprocess a source file
RicCorticalThicknessByNormal.cpp.i:
	$(MAKE) -f CMakeFiles/RicCorticalThicknessByNormal.dir/build.make CMakeFiles/RicCorticalThicknessByNormal.dir/RicCorticalThicknessByNormal.cpp.i
.PHONY : RicCorticalThicknessByNormal.cpp.i

RicCorticalThicknessByNormal.s: RicCorticalThicknessByNormal.cpp.s
.PHONY : RicCorticalThicknessByNormal.s

# target to generate assembly for a file
RicCorticalThicknessByNormal.cpp.s:
	$(MAKE) -f CMakeFiles/RicCorticalThicknessByNormal.dir/build.make CMakeFiles/RicCorticalThicknessByNormal.dir/RicCorticalThicknessByNormal.cpp.s
.PHONY : RicCorticalThicknessByNormal.cpp.s

TexFill.o: TexFill.cpp.o
.PHONY : TexFill.o

# target to build an object file
TexFill.cpp.o:
	$(MAKE) -f CMakeFiles/RicCorticalThicknessByNormal.dir/build.make CMakeFiles/RicCorticalThicknessByNormal.dir/TexFill.cpp.o
.PHONY : TexFill.cpp.o

TexFill.i: TexFill.cpp.i
.PHONY : TexFill.i

# target to preprocess a source file
TexFill.cpp.i:
	$(MAKE) -f CMakeFiles/RicCorticalThicknessByNormal.dir/build.make CMakeFiles/RicCorticalThicknessByNormal.dir/TexFill.cpp.i
.PHONY : TexFill.cpp.i

TexFill.s: TexFill.cpp.s
.PHONY : TexFill.s

# target to generate assembly for a file
TexFill.cpp.s:
	$(MAKE) -f CMakeFiles/RicCorticalThicknessByNormal.dir/build.make CMakeFiles/RicCorticalThicknessByNormal.dir/TexFill.cpp.s
.PHONY : TexFill.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... RicCorticalThicknessByNormal"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... GM_Normal.o"
	@echo "... GM_Normal.i"
	@echo "... GM_Normal.s"
	@echo "... RicCorticalThicknessByNormal.o"
	@echo "... RicCorticalThicknessByNormal.i"
	@echo "... RicCorticalThicknessByNormal.s"
	@echo "... TexFill.o"
	@echo "... TexFill.i"
	@echo "... TexFill.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system


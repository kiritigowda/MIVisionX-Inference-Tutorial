# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_SOURCE_DIR = /home/lakshmi/Downloads/lk/MIVisionX-Inference-Tutorial

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lakshmi/Downloads/lk/MIVisionX-Inference-Tutorial

# Include any dependencies generated for this target.
include CMakeFiles/caffeModels.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/caffeModels.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/caffeModels.dir/flags.make

CMakeFiles/caffeModels.dir/source/annmodule.cpp.o: CMakeFiles/caffeModels.dir/flags.make
CMakeFiles/caffeModels.dir/source/annmodule.cpp.o: source/annmodule.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lakshmi/Downloads/lk/MIVisionX-Inference-Tutorial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/caffeModels.dir/source/annmodule.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/caffeModels.dir/source/annmodule.cpp.o -c /home/lakshmi/Downloads/lk/MIVisionX-Inference-Tutorial/source/annmodule.cpp

CMakeFiles/caffeModels.dir/source/annmodule.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/caffeModels.dir/source/annmodule.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lakshmi/Downloads/lk/MIVisionX-Inference-Tutorial/source/annmodule.cpp > CMakeFiles/caffeModels.dir/source/annmodule.cpp.i

CMakeFiles/caffeModels.dir/source/annmodule.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/caffeModels.dir/source/annmodule.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lakshmi/Downloads/lk/MIVisionX-Inference-Tutorial/source/annmodule.cpp -o CMakeFiles/caffeModels.dir/source/annmodule.cpp.s

CMakeFiles/caffeModels.dir/source/annmodule.cpp.o.requires:

.PHONY : CMakeFiles/caffeModels.dir/source/annmodule.cpp.o.requires

CMakeFiles/caffeModels.dir/source/annmodule.cpp.o.provides: CMakeFiles/caffeModels.dir/source/annmodule.cpp.o.requires
	$(MAKE) -f CMakeFiles/caffeModels.dir/build.make CMakeFiles/caffeModels.dir/source/annmodule.cpp.o.provides.build
.PHONY : CMakeFiles/caffeModels.dir/source/annmodule.cpp.o.provides

CMakeFiles/caffeModels.dir/source/annmodule.cpp.o.provides.build: CMakeFiles/caffeModels.dir/source/annmodule.cpp.o


# Object files for target caffeModels
caffeModels_OBJECTS = \
"CMakeFiles/caffeModels.dir/source/annmodule.cpp.o"

# External object files for target caffeModels
caffeModels_EXTERNAL_OBJECTS =

libcaffeModels.so: CMakeFiles/caffeModels.dir/source/annmodule.cpp.o
libcaffeModels.so: CMakeFiles/caffeModels.dir/build.make
libcaffeModels.so: CMakeFiles/caffeModels.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lakshmi/Downloads/lk/MIVisionX-Inference-Tutorial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libcaffeModels.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/caffeModels.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/caffeModels.dir/build: libcaffeModels.so

.PHONY : CMakeFiles/caffeModels.dir/build

CMakeFiles/caffeModels.dir/requires: CMakeFiles/caffeModels.dir/source/annmodule.cpp.o.requires

.PHONY : CMakeFiles/caffeModels.dir/requires

CMakeFiles/caffeModels.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/caffeModels.dir/cmake_clean.cmake
.PHONY : CMakeFiles/caffeModels.dir/clean

CMakeFiles/caffeModels.dir/depend:
	cd /home/lakshmi/Downloads/lk/MIVisionX-Inference-Tutorial && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lakshmi/Downloads/lk/MIVisionX-Inference-Tutorial /home/lakshmi/Downloads/lk/MIVisionX-Inference-Tutorial /home/lakshmi/Downloads/lk/MIVisionX-Inference-Tutorial /home/lakshmi/Downloads/lk/MIVisionX-Inference-Tutorial /home/lakshmi/Downloads/lk/MIVisionX-Inference-Tutorial/CMakeFiles/caffeModels.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/caffeModels.dir/depend


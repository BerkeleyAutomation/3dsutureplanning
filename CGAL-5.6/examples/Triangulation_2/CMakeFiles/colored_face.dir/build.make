# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/virajramakrishnan/Desktop/Uni/AUTOLab/CGAL-5.6/examples/Triangulation_2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/virajramakrishnan/Desktop/Uni/AUTOLab/CGAL-5.6/examples/Triangulation_2

# Include any dependencies generated for this target.
include CMakeFiles/colored_face.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/colored_face.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/colored_face.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/colored_face.dir/flags.make

CMakeFiles/colored_face.dir/colored_face.cpp.o: CMakeFiles/colored_face.dir/flags.make
CMakeFiles/colored_face.dir/colored_face.cpp.o: colored_face.cpp
CMakeFiles/colored_face.dir/colored_face.cpp.o: CMakeFiles/colored_face.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/virajramakrishnan/Desktop/Uni/AUTOLab/CGAL-5.6/examples/Triangulation_2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/colored_face.dir/colored_face.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/colored_face.dir/colored_face.cpp.o -MF CMakeFiles/colored_face.dir/colored_face.cpp.o.d -o CMakeFiles/colored_face.dir/colored_face.cpp.o -c /Users/virajramakrishnan/Desktop/Uni/AUTOLab/CGAL-5.6/examples/Triangulation_2/colored_face.cpp

CMakeFiles/colored_face.dir/colored_face.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/colored_face.dir/colored_face.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/virajramakrishnan/Desktop/Uni/AUTOLab/CGAL-5.6/examples/Triangulation_2/colored_face.cpp > CMakeFiles/colored_face.dir/colored_face.cpp.i

CMakeFiles/colored_face.dir/colored_face.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/colored_face.dir/colored_face.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/virajramakrishnan/Desktop/Uni/AUTOLab/CGAL-5.6/examples/Triangulation_2/colored_face.cpp -o CMakeFiles/colored_face.dir/colored_face.cpp.s

# Object files for target colored_face
colored_face_OBJECTS = \
"CMakeFiles/colored_face.dir/colored_face.cpp.o"

# External object files for target colored_face
colored_face_EXTERNAL_OBJECTS =

colored_face: CMakeFiles/colored_face.dir/colored_face.cpp.o
colored_face: CMakeFiles/colored_face.dir/build.make
colored_face: /opt/homebrew/lib/libgmpxx.dylib
colored_face: /opt/homebrew/lib/libmpfr.dylib
colored_face: /opt/homebrew/lib/libgmp.dylib
colored_face: CMakeFiles/colored_face.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/virajramakrishnan/Desktop/Uni/AUTOLab/CGAL-5.6/examples/Triangulation_2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable colored_face"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/colored_face.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/colored_face.dir/build: colored_face
.PHONY : CMakeFiles/colored_face.dir/build

CMakeFiles/colored_face.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/colored_face.dir/cmake_clean.cmake
.PHONY : CMakeFiles/colored_face.dir/clean

CMakeFiles/colored_face.dir/depend:
	cd /Users/virajramakrishnan/Desktop/Uni/AUTOLab/CGAL-5.6/examples/Triangulation_2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/virajramakrishnan/Desktop/Uni/AUTOLab/CGAL-5.6/examples/Triangulation_2 /Users/virajramakrishnan/Desktop/Uni/AUTOLab/CGAL-5.6/examples/Triangulation_2 /Users/virajramakrishnan/Desktop/Uni/AUTOLab/CGAL-5.6/examples/Triangulation_2 /Users/virajramakrishnan/Desktop/Uni/AUTOLab/CGAL-5.6/examples/Triangulation_2 /Users/virajramakrishnan/Desktop/Uni/AUTOLab/CGAL-5.6/examples/Triangulation_2/CMakeFiles/colored_face.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/colored_face.dir/depend


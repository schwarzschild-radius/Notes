<span style="font-size:18px">

# Notes from the [Video](https://www.youtube.com/watch?v=6zOpxAwYKUQ)

## What is a build system?

* Orchestrates building your project.
* Should be efficient, only build the project that has been modified and not the whole system.
* Can also try to detect properties of the target system and customize your programs accordingly.
* Can build your project based on a flag to turn on/off portions of your build.
* Can also orchestrate testing of your project.

## Why should we use?
* Build your project asap.
* Project to be portable.
* Writing build system is tedious as your project sieze grows

## Existing Build System
* Handwritten makefiles
  * targets and build rules.
  * good for small project but tedious when the project size grows.
  * No native windows support
  * Not meant for configuration detection.
  * builds only specified part of the project that has been modified.

* Hand written Makefiles + Configure
  * Configure is a shell script used to detect properties of the target system. enable/disable features based on the system configuration(like compilers features, language support, hardware specs)
  * Hard to maintain.
  * There is a solution to the problem.
  * recursive makefiles are slow because they create a lot of make processes to parse each makefile.

* Autotools
  * Made up of a selection of tools(e.g. autoconf, automake, autoheader, ...)
  * Autoconf uses M4 macros to generate configure scripts.
  * Automake uses M4 macros to generate makefiles.
  * uses bash shell, M4 macros, perl...
  * No native Windows support.
  * Used by GNU projects.

* SCons
  * Uses to python to build project
  * Native windows support.

* Ninja
  * Similar to make.
  * It is make done right and has only targets and their rules and nothing else.
  * Non recursive, so it is fast.
  * Cross platform.

## Cmake

* It is a meta-build system
* It can be used to generate other build systems.
* Cross platform.
* Project Description(CMakeLists.txt) --> CMake --> othre build systems
* Other build systems include
  * Makefile project.
  * Eclipse project.
  * XCode project.
  * Ninja project.
  * Visual studio projects.
  * Other build system can be integrated using extensions.

## Who uses CMake?

* VTK / ITK library(original users of CMake).
* LLVM/Clang compiler framework.
* KDE desktop.
* MySQL(and its famous fork MariaDB).
* OpenCV library.
* Blender
* and many other.

## Ways of Invoking CMake
* CMake Command line tool
* ccmake Command line User Interface using ncurses.
* cmake-gui

## Building out of source
* This means having all generated binaries in a different folder from the source folder.
* Easy build for multiple configurations.
* Easy to clean builds
* CMake supports this out of the box.

  

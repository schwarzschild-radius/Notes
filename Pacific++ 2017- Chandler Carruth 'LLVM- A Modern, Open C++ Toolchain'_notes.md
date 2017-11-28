<span style="font-size:18px">

# Notes from Talk

* C++ is taking off

* But not the **Tool chains are not**

## Distribution support

* Debian: GCC 6, clang 3.8
* Ubuntu LTS: GCC 5, clang 3.8
* RHEL : GCC 4.8, clang: Nope!

## What happens with the bug

* Stacked release cycles.
  * wait the tool chain to update.
  * Wait for the distro to update.

* similarly find the bug and report the bug.
* wait for the next release of the toolchain.
* wait for the fixed toolchain to be available.


## Solution

* Need to develop toolchain in-house?
* It is hard to develop.
* Many are closed source and proprietary.
* But open-source compilers are hard to hack on.
* need to build binutils, GDB, etc.
* Debug miscompiles

# **_Impossible!_**

## What is llvm?

* Open source moduler collection of compiler & toolchain infrastructure.

* includes code generation, optimization, linking and loading.

* Written in C++.

* clang is a front-end and there are other frontends.

## What can you do with LLVM?

* code generation for DSLs
* traditional copilers for general purpose languages.
* Developer or editor tools for static analysis, refactoring

LLVM has bugs!

* Since LLVM is written in C++ so one can hack on LLVM to contribute for opensource community.

                    

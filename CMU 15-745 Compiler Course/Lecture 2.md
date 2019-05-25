# LLVM Compiler
* It is an infrastructure for building compilers.

## LLVM Compiler Architecture
* LLVM at it's core is an intermediate representation(IR)
* Optimizations are implemented as a series of passes(analysis or tranformation) that acts on the IR and produces a transformed IR.
* LLVM IR is a virutal instruction which is also strongly typed

## Steps in Clang Compilation
1. C/C++ source code - clang
2. Clang AST - clang, clang-tidy
3. LLVM IR - opt
4. SelectionDAG - llc
5. MachineInst - llc
6. MCInst / Assembly -llc
7. Link Time Optimizations - lto

## Goals of LLVM IR
1. Language and Target independent representation
2. High and Low level optimizations
3. Easy to understand
4. Support for High and Low end optimiations

## Overview of the LLVM instruction set
1. RISC like three address code
2. Infinite virutal register set
3. Simple, low-level control flow constructs
4. Load/Store instructions with typed pointers
5. Explicit dataflow through SSA form
6. Explicit control flow graph even for exceptions
7. Explicit typed pointer arithmetic
8. Preserves array subscript and indexing
9. Lowering Higher language constructs to LLVM IR is easy

## LLVM Program structure
1. Module - Functions and Global variables
2. Function - Basic blocks and arguments
3. Basic blocs - List of instructions
4. Instruction - opcode + vector of instructions and all instructions are typed. All instructions have type.

## Traversing over LLVM Program Structure
1. LLVM API provides iterators to traverse through Modules and functions
2. Traversals occur through doubly linked lists


## LLVM Pass Manager
1. LLVM optimization are done through passes.
2. There are two types of passes
    1. Analysis pass - non-mutating pass
    2. Transformation pass - mutating pass
3. Each pass can depend on previous pass
4. There are six useful types of passes
    1. BasicBlockPass: iterate over all basic blocks
    2. CallGraphSCC: iterate over SCCs, in bottom up call graph order
    3. FunctionPass: iterate over all functions in particular order
    4. LoopPass: iterate over all loop in reverse nested loop order
    5. ModulePass: general interprocedural pass over program
    6. RegionPass: iterate over single-entry/exit regions, in reverse nester order
5. Custom passes are loaded using shared library.
6. Passes register themselves but custom passes must be resgistered by the user.



# Introduction to TLA+
- TLA+ enables high level modelling of the digital systems
    - high level: Above the code level and at the design level
    - digital systems: Algorithms, Computer System

- TLA+ abstracts away the implmentation details by describing only the necessary parts of the system for proper functioning

> Abstraction: Simplifiying by removing irrelevant details. Abstraction helps to engineer complex systems

- TLA+ helps one to be better at `Abstraction`
- TLA+ helps to represent an algorithm as a high-level specification
- Designing at a higher level can be helpful because it can expressed more easily where direct one to one mapping to code is not available
- TLA+ provides set of tools to check our design
    - `TLC - TLA+ model checker`, checks a given predicate on the `individual behavior` of the spec
-  But TLA+ cannot verify across individual behavior of the spec because it requires every execution to respect the predicate

## State Machines
- TLA+ Represents program execution as a sequence of discrete steps
    - Sequence of discrete steps means one possible `behavior` of the program
    - Each step is represented as state change of the `State Machine`
    - A `State Change` is representated in TLA+ as an assignment to variables of the system

- There are different ways of specifiying a digital system
    - Programming Languages
    - Turing Machines
    - Automata
    - HDLs
- But the above representations can be represented using `State Machines`

## Description of a State Machine
- A State Machine is represented using the following
    - All possible `Initial State`
    - Set of all possible `Next State` given the current state
    - It `halts` if there are not next state for the current state

- Since state change is represeted as  `assignments to variables`
    - All possible `Initial States` represents set of all possible value of the variables at start
    - Set of all next state given the current state is possible assignments to atleast one of those variables

- Eventually State Machines are much nicer to represent than programs because they capture the necessary details to reason about the program behavior

# Algebraic Data Types
Video Link: https://www.youtube.com/watch?v=OJzmWqCCZaM&list=WL&index=7
* Discussed Types
	- pair
	- tuple
	- optional
	- variant
* Algebraic Types - Composite type formed from combinig other types
* it is about the range of values a type can take
* Pair and Tuple are product types - the values are cartesian product of the value of the individual type
* In memory, pair and tuple is similar to POD type
	* May contain padding, alignment
	* Different std library can vary in the ABI
* Variant is a sum type. It is a typed union. The values of variant is the sum of the possible values of the constituent types
	* Only one type exists at any moment in time
* Algebraic Types
	* Sum Types
	* Product Types
* Variant is type safe
* Variant has an additional index field to query which type is active
* optional is another sum type which extends the type with a null value. The total possible value is n(vlaues) + 1
* std::any is not an algebraic types. It is a type erasure type. Takes any copyable types

Motivation
- optional is not used in std library
- optional can be used to avoid dynamic allocation for expression nullable type using pointer
- variant has a performance penalty compared to union because of the additional type safety mechanism
- Variant and optional allows in place construction of type using initialization
- using std::in_place for advanced in place construction
- std::get maps index to value
	- std::variant has only a partional mapping
- CTAD is supported in C++17
- std::variant is compared as {index(), values()}

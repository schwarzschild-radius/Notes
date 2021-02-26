# From STL to Ranges
- Ranges are re-design of STL
- Ranges is mix of
	- Concepts
	- Algorithms
	- Views
	- Range Adaptors
- Ranges change guarantees of some algorithms, so they are put in a new namspace

## Range
- It can be considered a iterator pair {begin, end}
- technically an iterator and sentinel
- sentinel and iterator can be different types
- Views can now be logically infinite


References:
- Jeff Garland, From STL to Ranges: Using Ranges Effectively, Cppcon 2019. [video](https://www.youtube.com/watch?v=vJ290qlAbbw&list=PLHTh1InhhwT6KhvViwRiTR7I5s09dLCSw&index=7) [slide](https://github.com/CppCon/CppCon2019/blob/master/Presentations/cpp20_standard_library_beyond_ranges/cpp20_standard_library_beyond_ranges__jeff_garland__cppcon_2019.pdf)

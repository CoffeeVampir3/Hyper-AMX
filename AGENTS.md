Modules (.cppm) are in ./modules
Tests (.cppm) are in ./tests
Benchmarks (.cppm) are in ./benchmarks
Documentation (.md) is in ./Documentation

CMakeLists doesn't need to be updated for tests and modules, it automatically adds tests and modules.
Don't attempt to build the project. The build is done on a remote server you cannot access.

Don't use git at all.

This project is using modern C++23.
Mandatory code references, always read before doing any code:
[ModernCPPGuide](Documentation/ModernCPPGuide.md) project code guidelines
[CPPModules](Documentation/CPPModules.md) for instructions on how to use modern C++ modules.

Domain References:
[AMX Guide](Documentation/AMXGuide.md)
[AVX512](Documentation/AVX512.md)
[TensorParallel](Documentation/TensorParallel.md)

# Code style
- Favor concise, self-explanatory code
- Avoid comments unless explictly told to do them.
- Avoid unnecessary line breaks
- Empty lines should not have any space

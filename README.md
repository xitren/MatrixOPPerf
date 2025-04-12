# General Matrix Multiplication (GeMM) cores 
Edicational code of GeMM cores with different types of optimizations using:
- Writing code better for optimizer (blocking)
- AVX256
- AVX512
- OpenMP + AVX512

Look at the results in:
clang_matrix_perf_test.txt
gcc_matrix_perf_test.txt

## Contents

- [Building and developing](#building-and-developing)
- [Project layout](#project-layout)
- [Contributing](#contributing)
- [Licensing](#licensing)

## Building and developing

The recommended way to develop and build this project is to use the Docker image as a dev container.

### Presets

This project makes use of [CMake
presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) to simplify the
process of configuring the project. As a developer, you should use a
`CMakePresets.json` file at the top-level directory of the repository.

### Configure, build and test

You can configure, build and test the `clang_host_release_linux` parts of the project with the following
commands:

~~~shell
cmake --preset=clang_host_release_linux
cmake --build --preset=clang_host_release_linux -t test
~~~

## Project layout

The following ideas are mainly stolen from [P1204R0 – Canonical Project
Structure](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1204r0.html).

- Also, all includes, even the "project local" ones use `<>` instead of `""`.
- Subfolders for the source code should comprise somewhat standalone "components".
- There should be a STATIC or INTERFACE library target for each component (this should
  make linking source code dependencies for tests easier).
- Everything test related is in `tests/` and its subdirectories.
- Tests have the `.cpp` extension and are
  named after the class, file, functionality, interface or whatever else they test.
- Hardware tests should be similar to unit tests and check simple functionalities of
  low-level code.
- Golden Tests are used for high level integration/system tests.

The following shows what the directory structure could actually look like.

<details>
  <summary>Directory structure</summary>

  ~~~
  patterns/
  ├── .github/
  ├── docs/
  ├── include/
  │   ├── xitren/
  │   │   ├── math/
  │   │   │   ├── branchless.hpp
  │   │   │   ├── detector.hpp
  │   │   │   ├── gemm_core.hpp
  │   │   │   └── matrix_alignment.hpp
  │   │   ├── simd/
  │   │   │   ├── gemm_double_simd.hpp
  │   │   │   ├── gemm_float_simd.hpp
  │   │   │   └── gemm_int8_simd.hpp
  │   │   └── ...
  |   └── ...
  │
  ├── tests/
  │   ├── custom/
  │   │   ├── CMakeLists.txt
  │   │   └── matrix_perf_test.hpp
  │   ├── CMakeLists.txt
  │   ├── math_branchless.cpp
  │   ├── math_gemm_core.cpp
  │   ├── math_gemm_double.cpp
  │   ├── math_gemm_float.cpp
  │   ├── math_gemm_int8.cpp
  │   ├── math_matrix_test.cpp
  │   ├── math_quant_perf_test.cpp
  │   └── ...
  │
  ├── .clang-format
  ├── .clang-tidy
  ├── .gitignore
  ├── CMakeLists.txt
  ├── CMakePresets.json
  ├── LICENSE
  ├── README.md
  └── ...
  ~~~

</details>

## Contributing

The best chance of getting a problem fixed is to submit a patch that fixes it (along with a test case that verifies the fix)!
Feel free to create PR.

## Licensing

See the [LICENSE](LICENSE) document.

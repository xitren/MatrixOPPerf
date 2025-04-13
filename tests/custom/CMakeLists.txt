cmake_minimum_required(VERSION 3.16)

message(STATUS "Adding test perfomance test -O0")
add_executable("Perf_O0" "matrix_perf_test.cpp")
target_compile_features("Perf_O0" PUBLIC cxx_std_20)
target_compile_options("Perf_O0" PRIVATE -Wall -Wextra -Wpedantic -Wc++20-compat -Wno-format-security
        -Woverloaded-virtual -Wsuggest-override -fopenmp -march=native -DEIGEN_DONT_VECTORIZE
        -O0 -fargument-noalias)
target_link_libraries("Perf_O0" PRIVATE ${LIBRARY_NAME} GTest::gtest GTest::GTest Eigen3::Eigen OpenMP::OpenMP_CXX -pthread)
gtest_discover_tests("Perf_O0" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS "Adding test perfomance test -Ofast")
add_executable("Perf_Ofast1" "matrix_perf_test.cpp")
target_compile_features("Perf_Ofast1" PUBLIC cxx_std_20)
target_compile_options("Perf_Ofast1" PRIVATE -Wall -Wextra -Wpedantic -Wc++20-compat -Wno-format-security
        -Woverloaded-virtual -Wsuggest-override -fopenmp -march=native
        -Ofast)
target_link_libraries("Perf_Ofast1" PRIVATE ${LIBRARY_NAME} GTest::gtest GTest::GTest Eigen3::Eigen OpenMP::OpenMP_CXX -pthread)
gtest_discover_tests("Perf_Ofast1" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

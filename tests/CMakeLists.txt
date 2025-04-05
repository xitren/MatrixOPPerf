cmake_minimum_required(VERSION 3.16)

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.11.0)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

FetchContent_Declare(
    Eigen3
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_LEAVE_TEST_IN_ALL_TARGET OFF)
set(EIGEN_BUILD_DOC OFF)
set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(Eigen3)

find_package(OpenMP REQUIRED)

include(GoogleTest)
add_subdirectory(custom)
# add_subdirectory(cuda)

file(GLOB TESTS *.cpp)

foreach (file ${TESTS})
    get_filename_component(tgt ${file} NAME_WE)
    message(STATUS "Adding test \"${tgt}\"")
    add_executable(${tgt} ${file})
    target_compile_features(${tgt} PUBLIC cxx_std_20)
    if (NOT ${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
        target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic -Wc++20-compat -Wno-format-security
                -Woverloaded-virtual -Wsuggest-override -fopenmp -DEIGEN_DONT_VECTORIZE
                -O0 -fargument-noalias)
    else()
        target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic -Wc++20-compat -Wno-format-security
                -Woverloaded-virtual -Wsuggest-override -fopenmp -DEIGEN_DONT_VECTORIZE
                -O0 -fargument-noalias)
    endif ()
    target_link_libraries(${tgt} PRIVATE ${LIBRARY_NAME} GTest::gtest GTest::GTest Eigen3::Eigen OpenMP::OpenMP_CXX -pthread)
    gtest_discover_tests(${tgt} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endforeach ()

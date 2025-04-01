cmake_minimum_required(VERSION 3.16)

set(LIBRARY_NAME
    matrix_calc_lib
)

set(GLOBAL_NAMESPACE
    xitren
)

project(${LIBRARY_NAME}
	DESCRIPTION "Matrix perfomance library"
    VERSION 1.0.4
	LANGUAGES CXX
)

add_library(${LIBRARY_NAME} INTERFACE)
add_library(${GLOBAL_NAMESPACE}::${LIBRARY_NAME} ALIAS ${LIBRARY_NAME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
target_compile_options(${LIBRARY_NAME} INTERFACE -std=c++20 -Wc++20-compat -march=native)

target_include_directories(
        ${LIBRARY_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)

target_sources(${LIBRARY_NAME} INTERFACE )

target_link_libraries(${LIBRARY_NAME} INTERFACE)

message(STATUS "Adding library project \"${LIBRARY_NAME}\"")

enable_testing()
add_subdirectory(tests)

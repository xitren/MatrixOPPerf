#include <xitren/math/matrix_classic.hpp>
#include <xitren/simd/operations.h++>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <iostream>

using namespace xitren::simd;
using namespace xitren::math;

TEST(matrix_perf_test, usual)
{
    constexpr std::size_t size = 32;

    auto                               A = matrix_classic<double, size, size>::get_rand_matrix();
    auto                               B = matrix_classic<double, size, size>::get_rand_matrix();
    matrix_classic<double, size, size> C{{}};

    matrix_mult_basic_blocked(A, B, C);
}

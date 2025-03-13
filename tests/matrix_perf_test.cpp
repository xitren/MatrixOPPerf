#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <xitren/math/matrix_classic.hpp>
#include <xitren/simd/operations.h++>

using namespace xitren::simd;

TEST(matrix_perf_test, usual) {
    std::array<double, 4> A{{1, 2, 3, 4}};
    std::array<double, 4> B{{1, 2, 3, 4}};
    std::array<double, 4> C{{}};

    matrix_mult_unrolled(A, B, C);
}

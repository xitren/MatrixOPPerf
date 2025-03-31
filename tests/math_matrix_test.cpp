#include <xitren/math/matrix_strassen.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace xitren::math;

TEST(matrix_test, matrix_strassen_2x2)
{
    std::array<int, 4> A{{1, 3, 5, 7}};
    std::array<int, 4> B{{6, 8, 4, 2}};
    std::array<int, 4> C{{18, 14, 58, 54}};

    matrix_strassen<int, 2> mA{A};
    matrix_strassen<int, 2> mB{B};

    auto mC = mA * mB;

    EXPECT_EQ(C[0], mC.get(0, 0));
    EXPECT_EQ(C[1], mC.get(0, 1));
    EXPECT_EQ(C[2], mC.get(1, 0));
    EXPECT_EQ(C[3], mC.get(1, 1));
}

TEST(matrix_test, matrix_strassen_2x2_v2)
{
    std::array<int, 4> A{{1, 2, 6, 7}};
    std::array<int, 4> B{{10, 20, 40, 50}};
    std::array<int, 4> C{{90, 120, 340, 470}};

    matrix_strassen<int, 2> mA{A};
    matrix_strassen<int, 2> mB{B};

    auto mC = mA * mB;

    EXPECT_EQ(C[0], mC.get(0, 0));
    EXPECT_EQ(C[1], mC.get(0, 1));
    EXPECT_EQ(C[2], mC.get(1, 0));
    EXPECT_EQ(C[3], mC.get(1, 1));
}

TEST(matrix_test, matrix_strassen_4x4)
{
    std::array<int, 16> A{{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7}};
    std::array<int, 16> B{{7, 6, 5, 4, 3, 2, 1, 9, 8, 7, 6, 5, 4, 3, 2, 1}};
    std::array<int, 16> C{{53, 43, 33, 41, 141, 115, 89, 117, 94, 79, 64, 58, 119, 97, 75, 98}};

    matrix_strassen<int, 4> mA{A};
    matrix_strassen<int, 4> mB{B};

    auto mC = mA * mB;

    for (std::size_t l{}; l < 4; l++) {
        for (std::size_t m{}; m < 4; m++) {
            EXPECT_EQ(C[(l)*4 + m], mC.get(l, m));
        }
    }
}

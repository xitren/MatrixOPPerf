#include <xitren/math/branchless.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <iostream>
#include <limits>

using namespace xitren::math;

TEST(branchless_test, base)
{
    int a = 6;
    int b = 7;

    auto& sel = branchless_select(a < b, a, b);
    std::cout << sel << std::endl;
    EXPECT_EQ(sel, a);

    auto& sel2 = branchless_select(a > b, a, b);
    std::cout << sel2 << std::endl;
    EXPECT_EQ(sel2, b);
}

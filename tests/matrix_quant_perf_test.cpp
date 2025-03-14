#include <Eigen/Core>
#include <xitren/math/matrix_alignment.hpp>
#include <xitren/math/matrix_classic.hpp>
#include <xitren/math/matrix_strassen.hpp>
#include <xitren/simd/operations.h++>

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace xitren::math;
using namespace xitren::simd;
using namespace std::literals;

using time_type = std::chrono::microseconds;

struct measurement {
    std::size_t us;
    std::size_t cycles;
};

auto
measure(std::function<std::size_t(void)> callback)
{
    using time       = std::chrono::high_resolution_clock;
    using fsec       = std::chrono::duration<float>;
    auto      t0     = time::now();
    auto      cycles = callback();
    auto      t1     = time::now();
    fsec      fs     = t1 - t0;
    time_type d      = std::chrono::duration_cast<time_type>(fs);
    return measurement{static_cast<std::size_t>(d.count()), cycles};
}

auto
matrix_test(std::string name, measurement base, std::function<void(void)> callback)
{
    auto calc_time = measure([&]() -> std::size_t {
        std::size_t cnt{};
        time_type   period_{1000000us};
        auto        start_time{std::chrono::system_clock::now()};
        auto        last_time{std::chrono::system_clock::now()};
        while ((last_time - start_time) <= period_) {
            for (std ::size_t i{}; i < 100; i++, cnt++) {
                callback();
            }
            last_time = std::chrono::system_clock::now();
        }
        return cnt;
    });
    if (base.us == 0) {
        std::cout << name << "\tTime:\t" << calc_time.us << "\tCycles:\t" << calc_time.cycles << "\tx1" << std::endl;
    } else {
        double const in{static_cast<double>(base.cycles) / static_cast<double>(base.us)};
        double const out{static_cast<double>(calc_time.cycles) / static_cast<double>(calc_time.us)};
        std::cout << name << "\tTime:\t" << calc_time.us << "\tCycles:\t" << calc_time.cycles << "\tx"
                  << static_cast<int>(out / in) << std::endl;
    }
    return calc_time;
}

template <std::size_t Size>
void
test_sized_int32_matrix()
{
    constexpr std::size_t size = Size;
    std::cout << "\nMatrix C = A * B size:\t" << size << std::endl;

    auto A = matrix_classic<std::int32_t, size, size>::get_rand_matrix(0., 1.);
    auto B = matrix_classic<std::int32_t, size, size>::get_rand_matrix(0., 1.);
    auto C = matrix_classic<std::int32_t, size, size>::get_rand_matrix(0., 1.);

    auto base = matrix_test("Naive   ", measurement{0, 0}, [&]() { A->mult(*B, *C); });

    using loc_type_eigen = Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>;
    auto         mAe     = loc_type_eigen::Random(size, size);
    auto         mBe     = loc_type_eigen::Random(size, size);
    std::int32_t it{};
    matrix_test("Eigen   ", base, [&]() {
        auto mCe = mAe * mBe;
        it += mCe(0, 0);
    });

    if constexpr (Size < 256) {
        static auto Ast = matrix_strassen<std::int32_t, size>::get_rand_matrix();
        static auto Bst = matrix_strassen<std::int32_t, size>::get_rand_matrix();
        static auto Cst = matrix_strassen<std::int32_t, size>::get_rand_matrix();
        matrix_test("Strassen", base, [&]() { Cst = Ast * Bst; });
    }
}

template <std::size_t Size>
void
test_sized_int8_matrix()
{
    constexpr std::size_t size = Size;
    std::cout << "\nMatrix C = A * B size:\t" << size << std::endl;

    auto A = matrix_classic<std::int8_t, size, size>::get_rand_matrix(0., 1.);
    auto B = matrix_classic<std::int8_t, size, size>::get_rand_matrix(0., 1.);
    auto C = matrix_classic<std::int8_t, size, size>::get_rand_matrix(0., 1.);

    auto base = matrix_test("Naive   ", measurement{0, 0}, [&]() { A->mult(*B, *C); });

    using loc_type_eigen = Eigen::Matrix<std::int8_t, Eigen::Dynamic, Eigen::Dynamic>;
    auto        mAe      = loc_type_eigen::Random(size, size);
    auto        mBe      = loc_type_eigen::Random(size, size);
    std::int8_t it{};
    matrix_test("Eigen   ", base, [&]() {
        auto mCe = mAe * mBe;
        it += mCe(0, 0);
    });

    if constexpr (Size < 256) {
        static auto Ast = matrix_strassen<std::int8_t, size>::get_rand_matrix();
        static auto Bst = matrix_strassen<std::int8_t, size>::get_rand_matrix();
        static auto Cst = matrix_strassen<std::int8_t, size>::get_rand_matrix();
        matrix_test("Strassen", base, [&]() { Cst = Ast * Bst; });
    }
}

TEST(matrix_quant_int32_perf_test, usual)
{
    test_sized_int32_matrix<4>();
    test_sized_int32_matrix<8>();
    test_sized_int32_matrix<16>();
    test_sized_int32_matrix<32>();
    test_sized_int32_matrix<64>();
    test_sized_int32_matrix<128>();
    test_sized_int32_matrix<256>();
}

TEST(matrix_quant_int8_perf_test, usual)
{
    test_sized_int8_matrix<4>();
    test_sized_int8_matrix<8>();
    test_sized_int8_matrix<16>();
    test_sized_int8_matrix<32>();
    test_sized_int8_matrix<64>();
    test_sized_int8_matrix<128>();
    test_sized_int8_matrix<256>();
}

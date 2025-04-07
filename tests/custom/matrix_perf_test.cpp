#include <Eigen/Core>
#include <xitren/math/matrix_alignment.hpp>
#include <xitren/math/matrix_classic.hpp>
#include <xitren/math/matrix_strassen.hpp>
#include <xitren/simd/gemm_double_simd.hpp>
#include <xitren/simd/gemm_float_simd.hpp>
#include <xitren/simd/gemm_int8_simd.hpp>

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

using namespace xitren::math;
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
            for (std ::size_t i{}; i < 1; i++, cnt++) {
                callback();
            }
            last_time = std::chrono::system_clock::now();
        }
        return cnt;
    });
    if (base.us == 0) {
        std::cout << name << "\tTime:\t" << calc_time.us << "\tCycles:\t" << calc_time.cycles
                  << "\tx1" << std::endl;
    } else {
        double const in{static_cast<double>(base.cycles) / static_cast<double>(base.us)};
        double const out{static_cast<double>(calc_time.cycles) / static_cast<double>(calc_time.us)};
        std::cout << name << "\tTime:\t" << calc_time.us << "\tCycles:\t" << calc_time.cycles
                  << "\tx" << static_cast<int>(out / in) << "."
                  << ((static_cast<int>(out * 10 / in)) % 10) << std::endl;
    }
    return calc_time;
}

template <class Type, std::size_t Size, optimization Optim>
auto
check(std::string name, auto base)
{
    auto Aal = matrix_aligned<Type, Size, Size, Optim>::get_rand_matrix(0., 1.);
    auto Bal = matrix_aligned<Type, Size, Size, Optim>::get_rand_matrix(0., 1.);
    auto Cal = matrix_aligned<Type, Size, Size, Optim>::get_zeros_matrix();

    return matrix_test(name, base,
                       [&]() { matrix_aligned<Type, Size, Size, Optim>::mult(*Aal, *Bal, *Cal); });
}

template <class Type, std::size_t Size>
void
test_sized_matrix()
{
    constexpr std::size_t size = Size;
    std::string           str  = "X ";
    if constexpr (std::is_same<Type, double>()) {
        str = "D  ";
    }
    if constexpr (std::is_same<Type, float>()) {
        str = "F  ";
    }
    if constexpr (std::is_same<Type, std::int8_t>()) {
        str = "I8 ";
    }
    auto sz = std::to_string(size);

    auto base = check<Type, Size, optimization::naive>(" "s + sz + "\t" + str + "Naive   ",
                                                       measurement{0, 0});
    check<Type, Size, optimization::blocked>(" "s + sz + "\t" + str + "Blocked ", base);
    check<Type, Size, optimization::avx256>(" "s + sz + "\t" + str + "AVX256  ", base);
    check<Type, Size, optimization::avx512>(" "s + sz + "\t" + str + "AVX512  ", base);
    check<Type, Size, optimization::openmp_avx512_blocked>(" "s + sz + "\t" + str + "OpenMP  ",
                                                           base);

    {
        using loc_type_eigen = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
        auto   mAe           = loc_type_eigen::Random(size, size);
        auto   mBe           = loc_type_eigen::Random(size, size);
        double it{};
        matrix_test(" "s + sz + "\t" + str + "Eigen   ", base, [&]() {
            auto mCe = mAe * mBe;
            it += mCe(0, 0);
        });
    }
}

TEST(matrix_perf_test, type)
{
    test_sized_matrix<float, 32>();
    test_sized_matrix<double, 32>();
    test_sized_matrix<std::int8_t, 64>();
    test_sized_matrix<float, 64>();
    test_sized_matrix<double, 64>();
    test_sized_matrix<std::int8_t, 128>();
    test_sized_matrix<float, 128>();
    test_sized_matrix<double, 128>();
    test_sized_matrix<std::int8_t, 256>();
    test_sized_matrix<float, 256>();
    test_sized_matrix<double, 256>();
    test_sized_matrix<std::int8_t, 512>();
    test_sized_matrix<float, 512>();
    test_sized_matrix<double, 512>();
    test_sized_matrix<std::int8_t, 1024>();
    test_sized_matrix<float, 1024>();
    test_sized_matrix<double, 1024>();
    test_sized_matrix<std::int8_t, 2048>();
    test_sized_matrix<float, 2048>();
    test_sized_matrix<double, 2048>();
}

TEST(matrix_perf_test, type_long)
{
    test_sized_matrix<std::int8_t, 4096>();
    test_sized_matrix<float, 4096>();
    test_sized_matrix<double, 4096>();
    test_sized_matrix<std::int8_t, 8192>();
    test_sized_matrix<float, 8192>();
    test_sized_matrix<double, 8192>();
    test_sized_matrix<std::int8_t, 16384>();
    test_sized_matrix<float, 16384>();
    test_sized_matrix<double, 16384>();
}

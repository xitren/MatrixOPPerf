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
#include <sstream>
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
matrix_test(std::string format, std::string optim, std::size_t size, measurement base,
            std::ofstream& outfile, std::function<void(void)> callback)
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
    auto gops      = (3 * size * size * size * calc_time.cycles) / (calc_time.us * 1'000);
    outfile << format << ";" << optim << ";" << calc_time.us << ";" << calc_time.cycles << ";"
            << gops << std::endl;
    if (base.us == 0) {
        std::cout << format << "\t" << optim << "\tTime:\t" << calc_time.us << "\tCycles:\t"
                  << calc_time.cycles << "\tGIPS:\t" << gops << "\tx1" << std::endl;
    } else {
        double const in{static_cast<double>(base.cycles) / static_cast<double>(base.us)};
        double const out{static_cast<double>(calc_time.cycles) / static_cast<double>(calc_time.us)};
        std::cout << format << "\t" << optim << "\tTime:\t" << calc_time.us << "\tCycles:\t"
                  << calc_time.cycles << "\tGIPS:\t" << gops << "\tx" << static_cast<int>(out / in)
                  << "." << ((static_cast<int>(out * 10 / in)) % 10) << std::endl;
    }
    return calc_time;
}

template <class Type, std::size_t Size, optimization Optim>
auto
check(std::string format, std::string optim, measurement base, std::ofstream& outfile)
{
    auto Aal = matrix_aligned<Type, Size, Size, Optim>::get_rand_matrix(0., 1.);
    auto Bal = matrix_aligned<Type, Size, Size, Optim>::get_rand_matrix(0., 1.);
    auto Cal = matrix_aligned<Type, Size, Size, Optim>::get_zeros_matrix();

    return matrix_test(format, optim, Size, base, outfile,
                       [&]() { matrix_aligned<Type, Size, Size, Optim>::mult(*Aal, *Bal, *Cal); });
}

template <class Type, std::size_t Size>
void
test_sized_matrix(std::ofstream& outfile)
{
    constexpr std::size_t size = Size;
    std::string           str  = "X";
    if constexpr (std::is_same<Type, double>()) {
        str = "Double";
    }
    if constexpr (std::is_same<Type, float>()) {
        str = "Float";
    }
    if constexpr (std::is_same<Type, std::int8_t>()) {
        str = "INT8";
    }
    auto sz = std::to_string(size);

    auto base = check<Type, Size, optimization::naive>(sz, "Naive", measurement{0, 0}, outfile);
    check<Type, Size, optimization::blocked>(sz, "Blocked", base, outfile);
    check<Type, Size, optimization::avx256>(sz, "AVX256", base, outfile);
    check<Type, Size, optimization::avx512>(sz, "AVX512", base, outfile);
    check<Type, Size, optimization::openmp_avx512_blocked>(sz, "OpenMP", base, outfile);

    {
        using loc_type_eigen = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
        auto   mAe           = loc_type_eigen::Random(size, size);
        auto   mBe           = loc_type_eigen::Random(size, size);
        double it{};
        matrix_test(sz, "Eigen", Size, base, outfile, [&]() {
            auto mCe = mAe * mBe;
            it += mCe(0, 0);
        });
    }
}

std::ofstream outfile("test.csv");

template <class Type>
void
check_line(std::ofstream& outfile)
{
    test_sized_matrix<Type, 64>(outfile);
    test_sized_matrix<Type, 128>(outfile);
    test_sized_matrix<Type, 192>(outfile);
    test_sized_matrix<Type, 256>(outfile);
    test_sized_matrix<Type, 320>(outfile);
    test_sized_matrix<Type, 384>(outfile);
    test_sized_matrix<Type, 448>(outfile);
    test_sized_matrix<Type, 512>(outfile);
    test_sized_matrix<Type, 576>(outfile);
    test_sized_matrix<Type, 640>(outfile);
    test_sized_matrix<Type, 704>(outfile);
    test_sized_matrix<Type, 768>(outfile);
    test_sized_matrix<Type, 832>(outfile);
    test_sized_matrix<Type, 896>(outfile);
    test_sized_matrix<Type, 960>(outfile);
    test_sized_matrix<Type, 1024>(outfile);
}

TEST(matrix_perf_test, type)
{
    outfile << "Format;Optimization;Time (us);Cycles;GFLOPS/GIPS\n"s;

    check_line<std::int8_t>(outfile);
    check_line<float>(outfile);
    check_line<double>(outfile);
}

TEST(matrix_perf_test, type_long)
{
    test_sized_matrix<std::int8_t, 2048>(outfile);
    test_sized_matrix<float, 2048>(outfile);
    test_sized_matrix<double, 2048>(outfile);
    test_sized_matrix<std::int8_t, 4096>(outfile);
    test_sized_matrix<float, 4096>(outfile);
    test_sized_matrix<double, 4096>(outfile);
    test_sized_matrix<std::int8_t, 8192>(outfile);
    test_sized_matrix<float, 8192>(outfile);
    test_sized_matrix<double, 8192>(outfile);
    test_sized_matrix<std::int8_t, 16384>(outfile);
    test_sized_matrix<float, 16384>(outfile);
    test_sized_matrix<double, 16384>(outfile);
}

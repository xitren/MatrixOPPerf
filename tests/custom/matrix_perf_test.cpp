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
            for (std ::size_t i{}; i < 1; i++, cnt++) {
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
                  << static_cast<int>(out / in) << "." << ((static_cast<int>(out * 10 / in)) % 10) << std::endl;
    }
    return calc_time;
}

template <std::size_t Size>
void
test_sized_matrix()
{
    constexpr std::size_t size = Size;
    std::cout << "\nMatrix C = A * B size:\t" << size << std::endl;

    auto A = matrix_classic<double, size, size>::get_rand_matrix(0., 1.);
    auto B = matrix_classic<double, size, size>::get_rand_matrix(0., 1.);
    auto C = matrix_classic<double, size, size>::get_rand_matrix(0., 1.);

    auto base = matrix_test("Naive   ", measurement{0, 0}, [&]() { A->mult(*B, *C); });
    matrix_test("Blocked ", base, [&]() { matrix_mult_basic_blocked(*A, *B, *C); });

    auto Aal = matrix_aligned<double, size, size>::get_rand_matrix(0., 1.);
    auto Bal = matrix_aligned<double, size, size>::get_rand_matrix(0., 1.);
    auto Cal = matrix_aligned<double, size, size>::get_rand_matrix(0., 1.);

    matrix_test("AVX256  ", base, [&]() { matrix_aligned<double, size, size>::mult_256(*Aal, *Bal, *Cal); });
    matrix_test("AVX512  ", base, [&]() { matrix_aligned<double, size, size>::mult_512(*Aal, *Bal, *Cal); });
    matrix_test("Unrolled", base, [&]() { matrix_aligned<double, size, size>::mult_unrolled(*Aal, *Bal, *Cal); });
    matrix_test("OpenMP  ", base, [&]() { matrix_aligned<double, size, size>::mult_openmp(*Aal, *Bal, *Cal); });

    using loc_type_eigen = Eigen::MatrixXd;
    auto   mAe           = loc_type_eigen::Random(size, size);
    auto   mBe           = loc_type_eigen::Random(size, size);
    double it{};
    matrix_test("Eigen   ", base, [&]() {
        auto mCe = mAe * mBe;
        it += mCe(0, 0);
    });

    if constexpr (Size < 256) {
        static auto Ast = matrix_strassen<double, size>::get_rand_matrix();
        static auto Bst = matrix_strassen<double, size>::get_rand_matrix();
        static auto Cst = matrix_strassen<double, size>::get_rand_matrix();
        matrix_test("Strassen", base, [&]() { Cst = Ast * Bst; });
    }
}

TEST(matrix_perf_test, usual)
{
    test_sized_matrix<32>();
    test_sized_matrix<64>();
    test_sized_matrix<128>();
    test_sized_matrix<256>();
    test_sized_matrix<512>();
    test_sized_matrix<1024>();
    test_sized_matrix<2048>();
    test_sized_matrix<4096>();
    test_sized_matrix<8192>();
    test_sized_matrix<16384>();
    test_sized_matrix<32768>();
}

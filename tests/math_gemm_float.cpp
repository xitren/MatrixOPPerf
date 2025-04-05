#include <Eigen/Core>
#include <xitren/math/matrix_alignment.hpp>
#include <xitren/simd/gemm_float_simd.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace xitren::math;

template <class Type, std::size_t Rows, std::size_t Columns, optimization Alg>
void
print(matrix_aligned<Type, Rows, Columns, Alg>& rr)
{
    for (std::size_t i = 0; i < Rows; i++) {
        for (std::size_t j = 0; j < Columns; j++) {
            std::cout << rr.get(i, j) << "\t";
        }
        std::cout << std::endl;
    }
}

template <class Type, std::size_t Rows, std::size_t ColumnsOther, std::size_t Columns,
          optimization Alg>
void
check_with_eigen(matrix_aligned<Type, Rows, ColumnsOther, Alg>&    a,
                 matrix_aligned<Type, ColumnsOther, Columns, Alg>& b,
                 matrix_aligned<Type, Rows, Columns, Alg>&         c)
{
    using loc_type_eigen = Eigen::MatrixXf;
    loc_type_eigen mAe{};
    loc_type_eigen mBe{};
    mAe.resize(Rows, ColumnsOther);
    mBe.resize(ColumnsOther, Columns);

    for (std::size_t i = 0; i < Rows; i++) {
        for (std::size_t j = 0; j < ColumnsOther; j++) {
            mAe(i, j) = a.get(i, j);
        }
    }
    for (std::size_t i = 0; i < ColumnsOther; i++) {
        for (std::size_t j = 0; j < Columns; j++) {
            mBe(i, j) = b.get(i, j);
        }
    }

    auto mCe = mAe * mBe;
    std::cout << "mAe" << std::endl;
    std::cout << mAe << std::endl;
    std::cout << "\n" << std::endl;
    std::cout << "mBe" << std::endl;
    std::cout << mBe << std::endl;
    std::cout << "\n" << std::endl;
    std::cout << "mCe" << std::endl;
    std::cout << mCe << std::endl;
    std::cout << "\n" << std::endl;
    std::cout << "Cal" << std::endl;
    print(c);
    std::cout << "\n" << std::endl;

    for (std::size_t i = 0; i < Rows; i++) {
        for (std::size_t j = 0; j < Columns; j++) {
            ASSERT_NEAR(mCe(i, j), c.get(i, j), 0.0001);
        }
    }
}

TEST(gemm_float, avx256_quad_8)
{
    constexpr std::size_t  size  = 8;
    constexpr optimization optim = optimization::avx256;

    auto Aal = matrix_aligned<float, size, size, optim>::get_rand_matrix(0., 1.);
    auto Bal = matrix_aligned<float, size, size, optim>::get_rand_matrix(0., 1.);
    auto Cal = matrix_aligned<float, size, size, optim>::get_rand_matrix(0., 1.);

    for (std::size_t i = 0; i < size; i++) {
        for (std::size_t j = 0; j < size; j++) {
            Cal->get(i, j) = 0;
        }
    }

    matrix_aligned<float, size, size, optim>::mult(*Aal, *Bal, *Cal);

    check_with_eigen(*Aal, *Bal, *Cal);
}

TEST(gemm_float, avx256_16_32_16)
{
    constexpr std::size_t  sizeRows         = 16;
    constexpr std::size_t  sizeOtherColumns = 32;
    constexpr std::size_t  sizeColumns      = 16;
    constexpr optimization optim            = optimization::avx256;

    auto Aal = matrix_aligned<float, sizeRows, sizeOtherColumns, optim>::get_rand_matrix(0., 1.);
    auto Bal = matrix_aligned<float, sizeOtherColumns, sizeColumns, optim>::get_rand_matrix(0., 1.);
    auto Cal = matrix_aligned<float, sizeRows, sizeColumns, optim>::get_rand_matrix(0., 1.);

    for (std::size_t i = 0; i < sizeRows; i++) {
        for (std::size_t j = 0; j < sizeColumns; j++) {
            Cal->get(i, j) = 0;
        }
    }

    matrix_aligned<float, sizeRows, sizeColumns, optim>::mult(*Aal, *Bal, *Cal);

    check_with_eigen(*Aal, *Bal, *Cal);
}

TEST(gemm_float, avx512_16_32_16)
{
    constexpr std::size_t  sizeRows         = 16;
    constexpr std::size_t  sizeOtherColumns = 32;
    constexpr std::size_t  sizeColumns      = 16;
    constexpr optimization optim            = optimization::avx512;

    auto Aal = matrix_aligned<float, sizeRows, sizeOtherColumns, optim>::get_rand_matrix(0., 1.);
    auto Bal = matrix_aligned<float, sizeOtherColumns, sizeColumns, optim>::get_rand_matrix(0., 1.);
    auto Cal = matrix_aligned<float, sizeRows, sizeColumns, optim>::get_rand_matrix(0., 1.);

    for (std::size_t i = 0; i < sizeRows; i++) {
        for (std::size_t j = 0; j < sizeColumns; j++) {
            Cal->get(i, j) = 0;
        }
    }

    matrix_aligned<float, sizeRows, sizeColumns, optim>::mult(*Aal, *Bal, *Cal);

    check_with_eigen(*Aal, *Bal, *Cal);
}

TEST(gemm_float, openmp_avx512_blocked_64_128_64)
{
    constexpr std::size_t  sizeRows         = 64;
    constexpr std::size_t  sizeOtherColumns = 128;
    constexpr std::size_t  sizeColumns      = 64;
    constexpr optimization optim            = optimization::openmp_avx512_blocked;

    auto Aal = matrix_aligned<float, sizeRows, sizeOtherColumns, optim>::get_rand_matrix(0., 1.);
    auto Bal = matrix_aligned<float, sizeOtherColumns, sizeColumns, optim>::get_rand_matrix(0., 1.);
    auto Cal = matrix_aligned<float, sizeRows, sizeColumns, optim>::get_rand_matrix(0., 1.);

    for (std::size_t i = 0; i < sizeRows; i++) {
        for (std::size_t j = 0; j < sizeColumns; j++) {
            Cal->get(i, j) = 0;
        }
    }

    matrix_aligned<float, sizeRows, sizeColumns, optim>::mult(*Aal, *Bal, *Cal);
    std::cout << "Cal" << std::endl;
    print(*Cal);
    std::cout << "\n" << std::endl;

    check_with_eigen(*Aal, *Bal, *Cal);
}

#pragma once

#include <xitren/simd/operations.h++>

#include <x86intrin.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

namespace xitren::math {

template <class Type, std::size_t Rows, std::size_t Columns>
class matrix_aligned {

public:
    matrix_aligned()
    {
        data_ = (Type*)_mm_malloc(Rows * Columns * sizeof(Type), 64);
        if (data_ == nullptr)
            throw std::bad_alloc{};    // ("failed to allocate largest problem size");
    }
    ~matrix_aligned() { _mm_free(data_); }

    static void
    mult_256(matrix_aligned const& A, matrix_aligned const& B, matrix_aligned& C)
    {
        static_assert((Rows & (Rows - 1)) == 0, "Should be power of 2!");
        static_assert(Rows == Columns);
        xitren::simd::matrix_mult_avx256(Rows, A.data_, B.data_, C.data_);
    }

    static void
    mult_512(matrix_aligned const& A, matrix_aligned const& B, matrix_aligned& C)
    {
        static_assert((Rows & (Rows - 1)) == 0, "Should be power of 2!");
        static_assert(Rows == Columns);
        xitren::simd::matrix_mult_avx512(Rows, A.data_, B.data_, C.data_);
    }

    static void
    mult_unrolled(matrix_aligned const& A, matrix_aligned const& B, matrix_aligned& C)
    {
        static_assert((Rows & (Rows - 1)) == 0, "Should be power of 2!");
        static_assert(Rows == Columns);
        xitren::simd::matrix_mult_avx512(Rows, A.data_, B.data_, C.data_);
    }

    static void
    mult_openmp(matrix_aligned const& A, matrix_aligned const& B, matrix_aligned& C)
    {
        static_assert((Rows & (Rows - 1)) == 0, "Should be power of 2!");
        static_assert(Rows == Columns);
        xitren::simd::matrix_mult_openmp(Rows, A.data_, B.data_, C.data_);
    }

    static std::shared_ptr<matrix_aligned>
    get_rand_matrix(double max_val, double min_val)
    {
        std::shared_ptr<matrix_aligned>  ret = std::make_shared<matrix_aligned>();
        std::random_device               rd;
        std::mt19937                     gen(rd());
        std::uniform_real_distribution<> dis(min_val, max_val);
        for (std::uint32_t i = 0; i < Rows * Columns; ++i) {
            ret->data_[i] = dis(gen);
        }
        return ret;
    }

private:
    double* data_{nullptr};
};

}    // namespace xitren::math

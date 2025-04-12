#pragma once
#include <xitren/math/gemm_core.hpp>

#include <x86intrin.h>

namespace xitren::math {

// This code snippet is defining a specialization of the `gemm_core` class template for the
// optimization strategy `optimization::avx256`.
template <std::uint_fast32_t Rows, std::uint_fast32_t Columns>
class gemm_core<Rows, Columns, double, optimization::avx256>
    : gemm_core<Rows, Columns, double, optimization::naive> {
    static constexpr std::uint_fast32_t vectorization = 4;
    static_assert(Columns >= vectorization, "Should be greater or equal to blocksize!");
    static_assert(!(Columns % vectorization), "Should be dividable to blocksize!");

public:
    using gemm_core<Rows, Columns, double, optimization::naive>::add;
    using gemm_core<Rows, Columns, double, optimization::naive>::sub;
    using gemm_core<Rows, Columns, double, optimization::naive>::transpose;
    using gemm_core<Rows, Columns, double, optimization::naive>::trace;
    using gemm_core<Rows, Columns, double, optimization::naive>::min;
    using gemm_core<Rows, Columns, double, optimization::naive>::max;

    template <std::uint_fast32_t Other>
    static void
    mult(double const* a, double const* b, double* c) noexcept
    {
        for (std::uint_fast32_t i = 0; i < Rows; i++) {
            for (std::uint_fast32_t j = 0; j < Columns; j += vectorization) {
                auto const current = i * Rows + j;
                __m256d    c0      = _mm256_load_pd(c + current); /* c0 = C[i][j] */
                for (std::uint_fast32_t k = 0; k < Other; k++) {
                    auto const var_a  = _mm256_broadcast_sd(a + k + i * Other);
                    auto const var_b  = _mm256_load_pd(b + j + k * Columns);
                    auto const mul_ab = _mm256_mul_pd(var_a, var_b);
                    c0                = _mm256_add_pd(c0, mul_ab); /* c0 += A[i][k]*B[k][j] */
                }
                _mm256_store_pd(c + current, c0);                  /* C[i][j] = c0 */
            }
        }
    }
};

// This code snippet is defining a specialization of the `gemm_core` class template for the
// optimization strategy `optimization::avx512`.
template <std::uint_fast32_t Rows, std::uint_fast32_t Columns>
class gemm_core<Rows, Columns, double, optimization::avx512>
    : gemm_core<Rows, Columns, double, optimization::naive> {
    static constexpr std::uint_fast32_t vectorization = 8;
    static_assert(Columns >= vectorization, "Should be greater or equal to blocksize!");
    static_assert(!(Columns % vectorization), "Should be dividable to blocksize!");

public:
    using gemm_core<Rows, Columns, double, optimization::naive>::add;
    using gemm_core<Rows, Columns, double, optimization::naive>::sub;
    using gemm_core<Rows, Columns, double, optimization::naive>::transpose;
    using gemm_core<Rows, Columns, double, optimization::naive>::trace;
    using gemm_core<Rows, Columns, double, optimization::naive>::min;
    using gemm_core<Rows, Columns, double, optimization::naive>::max;

    template <std::uint_fast32_t Other>
    static void
    mult(double const* a, double const* b, double* c) noexcept
    {
        for (std::uint_fast32_t i = 0; i < Rows; i++) {
            for (std::uint_fast32_t j = 0; j < Columns; j += vectorization) {
                auto const current = i * Columns + j;
                __m512d    c0      = _mm512_load_pd(c + current); /* c0 = C[i][j] */
                for (std::uint_fast32_t k = 0; k < Other; k++) {
                    auto const var_a = _mm512_broadcastsd_pd(_mm_load_sd(a + k + i * Other));
                    auto const var_b = _mm512_load_pd(b + j + k * Columns);
                    c0 = _mm512_fmadd_pd(var_a, var_b, c0); /* c0 += A[i][k]*B[k][j] */
                }
                _mm512_store_pd(c + current, c0);           /* C[i][j] = c0 */
            }
        }
    }
};

// This code snippet is defining a specialization of the `gemm_core` class template for the
// optimization strategy `optimization::openmp_avx512_blocked`.
template <std::uint_fast32_t Rows, std::uint_fast32_t Columns>
class gemm_core<Rows, Columns, double, optimization::openmp_avx512_blocked>
    : gemm_core<Rows, Columns, double, optimization::naive> {
    static constexpr std::uint_fast32_t blocksize     = 32;
    static constexpr std::uint_fast32_t vectorization = 8;
    static_assert(!(Rows % blocksize), "Should be dividable to blocksize!");
    static_assert(!(Columns % blocksize), "Should be dividable to blocksize!");

public:
    using gemm_core<Rows, Columns, double, optimization::naive>::add;
    using gemm_core<Rows, Columns, double, optimization::naive>::sub;
    using gemm_core<Rows, Columns, double, optimization::naive>::transpose;
    using gemm_core<Rows, Columns, double, optimization::naive>::trace;
    using gemm_core<Rows, Columns, double, optimization::naive>::min;
    using gemm_core<Rows, Columns, double, optimization::naive>::max;

    template <std::uint_fast32_t Other>
    static void
    mult(double const* a, double const* b, double* c) noexcept
    {
#pragma omp parallel for
        for (std::uint_fast32_t si = 0; si < Rows; si += blocksize) {
            for (std::uint_fast32_t sj = 0; sj < Columns; sj += blocksize) {
                for (std::uint_fast32_t sk = 0; sk < Other; sk += blocksize) {
                    do_block<Other>(si, sj, sk, a, b, c);
                }
            }
        }
    }

private:
    template <std::uint_fast32_t Other>
    static void
    do_block(const std::uint_fast32_t si, const std::uint_fast32_t sj, const std::uint_fast32_t sk,
             double const* a, double const* b, double* c) noexcept
    {
        auto const last_si = si + blocksize;
        auto const last_sj = sj + blocksize;
        auto const last_sk = sk + blocksize;
        for (std::uint_fast32_t i = si; i < last_si; ++i) {
            for (std::uint_fast32_t j = sj; j < last_sj; j += vectorization) {
                auto const current = i * Columns + j;
                __m512d    c0      = _mm512_load_pd(c + current); /* c0 = C[i][j] */
                for (std::uint_fast32_t k = sk; k < last_sk; ++k) {
                    auto const var_a = _mm512_broadcastsd_pd(_mm_load_sd(a + i * Other + k));
                    auto const var_b = _mm512_load_pd(b + k * Columns + j);
                    c0 = _mm512_fmadd_pd(var_a, var_b, c0); /* c0 += A[i][k]*B[k][j] */
                }
                _mm512_store_pd(c + current, c0);           /* C[i][j] = c0 */
            }
        }
    }
};

}    // namespace xitren::math

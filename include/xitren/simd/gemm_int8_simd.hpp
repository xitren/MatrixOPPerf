#pragma once
#include <xitren/math/gemm_core.hpp>

#include <x86intrin.h>

namespace xitren::math {

// This code snippet is defining a specialization of the `gemm_core` class template for the
// optimization strategy `optimization::avx256`.
template <std::uint_fast32_t Rows, std::uint_fast32_t Columns>
class gemm_core<Rows, Columns, std::int8_t, optimization::avx256>
    : gemm_core<Rows, Columns, std::int8_t, optimization::naive> {
    static constexpr std::uint_fast32_t vectorization = 32;
    static_assert(Columns >= vectorization, "Should be greater or equal to blocksize!");
    static_assert(!(Columns % vectorization), "Should be dividable to blocksize!");

public:
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::add;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::sub;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::transpose;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::trace;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::min;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::max;

    template <std::uint_fast32_t Other>
    static void
    mult(std::int8_t const* a, std::int8_t const* b, std::int8_t* c) noexcept
    {
        for (std::uint_fast32_t i = 0; i < Rows; i++) {
            for (std::uint_fast32_t j = 0; j < Columns; j += vectorization) {
                auto const current = i * Columns + j;
                __m256i    c0      = _mm256_loadu_epi8(c + current); /* c0 = C[i][j] */
                for (std::uint_fast32_t k = 0; k < Other; k++) {
                    auto const var_a = _mm256_broadcastb_epi8(_mm_loadu_epi8(a + i * Other + k));
                    auto const var_b = _mm256_loadu_epi8(b + k * Columns + j);

                    auto const dst_even = _mm256_mullo_epi16(var_a, var_b);
                    auto const dst_odd  = _mm256_mullo_epi16(_mm256_srli_epi16(var_a, 8),
                                                             _mm256_srli_epi16(var_b, 8));

                    auto const result
                        = _mm256_or_si256(_mm256_slli_epi16(dst_odd, 8),
                                          _mm256_and_si256(dst_even, _mm256_set1_epi16(0xFF)));
                    c0 = _mm256_add_epi8(c0, result); /* c0 += A[i][k]*B[k][j] */
                }
                _mm256_storeu_epi8(c + current, c0);  /* C[i][j] = c0 */
            }
        }
    }
};

// This code snippet is defining a specialization of the `gemm_core` class template for the
// optimization strategy `optimization::avx512`.
template <std::uint_fast32_t Rows, std::uint_fast32_t Columns>
class gemm_core<Rows, Columns, std::int8_t, optimization::avx512>
    : gemm_core<Rows, Columns, std::int8_t, optimization::naive> {
    static constexpr std::uint_fast32_t vectorization = 64;
    static_assert(Columns >= vectorization, "Should be greater or equal to blocksize!");
    static_assert(!(Columns % vectorization), "Should be dividable to blocksize!");

public:
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::add;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::sub;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::transpose;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::trace;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::min;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::max;

    template <std::uint_fast32_t Other>
    static void
    mult(std::int8_t const* a, std::int8_t const* b, std::int8_t* c) noexcept
    {
        for (std::uint_fast32_t i = 0; i < Rows; i++) {
            for (std::uint_fast32_t j = 0; j < Columns; j += vectorization) {
                auto const current = i * Columns + j;
                __m512i    c0      = _mm512_loadu_epi8(c + current); /* c0 = C[i][j] */
                for (std::uint_fast32_t k = 0; k < Other; k++) {
                    auto const var_a = _mm512_broadcastb_epi8(_mm_loadu_epi8(a + k + i * Other));
                    auto const var_b = _mm512_loadu_epi8(b + j + k * Columns);

                    auto const dst_even = _mm512_mullo_epi16(var_a, var_b);
                    auto const dst_odd  = _mm512_mullo_epi16(_mm512_srli_epi16(var_a, 8),
                                                             _mm512_srli_epi16(var_b, 8));

                    auto const result
                        = _mm512_or_si512(_mm512_slli_epi16(dst_odd, 8),
                                          _mm512_and_si512(dst_even, _mm512_set1_epi16(0xFF)));

                    c0 = _mm512_add_epi8(c0, result); /* c0 += A[i][k]*B[k][j] */
                }
                _mm512_storeu_epi8(c + current, c0);  /* C[i][j] = c0 */
            }
        }
    }
};

// This code snippet is defining a specialization of the `gemm_core` class template for the
// optimization strategy `optimization::openmp_avx512_blocked`.
template <std::uint_fast32_t Rows, std::uint_fast32_t Columns>
class gemm_core<Rows, Columns, std::int8_t, optimization::openmp_avx512_blocked>
    : gemm_core<Rows, Columns, std::int8_t, optimization::naive> {
    static constexpr std::uint_fast32_t blocksize     = 64;
    static constexpr std::uint_fast32_t vectorization = 64;
    static_assert(!(Rows % blocksize), "Should be dividable to blocksize!");
    static_assert(!(Columns % blocksize), "Should be dividable to blocksize!");

public:
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::add;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::sub;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::transpose;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::trace;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::min;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::max;

    template <std::uint_fast32_t Other>
    static void
    mult(std::int8_t const* a, std::int8_t const* b, std::int8_t* c) noexcept
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
             std::int8_t const* a, std::int8_t const* b, std::int8_t* c) noexcept
    {
        auto const last_si = si + blocksize;
        auto const last_sj = sj + blocksize;
        auto const last_sk = sk + blocksize;

        for (std::uint_fast32_t i = si; i < last_si; ++i) {
            for (std::uint_fast32_t j = sj; j < last_sj; j += vectorization) {
                auto const current = i * Columns + j;
                __m512i    c0      = _mm512_loadu_epi8(c + current); /* c0 = C[i][j] */
                for (std::uint_fast32_t k = sk; k < last_sk; ++k) {
                    auto const var_a = _mm512_broadcastb_epi8(_mm_loadu_epi8(a + i * Other + k));
                    auto const var_b = _mm512_loadu_epi8(b + k * Columns + j);

                    auto const dst_even = _mm512_mullo_epi16(var_a, var_b);
                    auto const dst_odd  = _mm512_mullo_epi16(_mm512_srli_epi16(var_a, 8),
                                                             _mm512_srli_epi16(var_b, 8));

                    auto const result
                        = _mm512_or_si512(_mm512_slli_epi16(dst_odd, 8),
                                          _mm512_and_si512(dst_even, _mm512_set1_epi16(0xFF)));

                    c0 = _mm512_add_epi8(c0, result); /* c0 += A[i][k]*B[k][j] */
                }
                _mm512_storeu_epi8(c + current, c0);  /* C[i][j] = c0 */
            }
        }
    }
};

}    // namespace xitren::math

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <utility>
#include <vector>

namespace xitren::math {

enum class optimization { naive, blocked, avx256, avx512, avx512_blocked, openmp_avx512_blocked };

template <std::uint_fast32_t Rows, std::uint_fast32_t Other, std::uint_fast32_t Columns, typename Type,
          optimization Alg>
class gemm_core {

public:
    static void
    mult(Type const* a, Type const* b, Type* c)
    {
        for (std::uint_fast32_t i = 0; i < Rows; ++i) {
            for (std::uint_fast32_t j = 0; j < Columns; ++j) {
                auto const current = i * Columns + j;
                Type       cij     = c[current];                  /* cij = C[i][j] */
                for (std::uint_fast32_t k = 0; k < Other; k++) {
                    cij += a[i * Other + k] * b[k * Columns + j]; /* cij += A[i][k]*B[k][j] */
                }
                c[current] = cij;                                 /* C[i][j] = cij */
            }
        }
    }
};

template <std::uint_fast32_t Rows, std::uint_fast32_t Other, std::uint_fast32_t Columns, typename Type>
class gemm_core<Rows, Other, Columns, Type, optimization::blocked> {
    static constexpr std::uint_fast32_t blocksize = 32;
    static_assert(!(Rows % blocksize), "Should be dividable to blocksize!");
    static_assert(!(Other % blocksize), "Should be dividable to blocksize!");
    static_assert(!(Columns % blocksize), "Should be dividable to blocksize!");

public:
    static void
    mult(Type const* a, Type const* b, Type* c)
    {
        for (std::uint_fast32_t si = 0; si < Rows; si += blocksize) {
            for (std::uint_fast32_t sj = 0; sj < Columns; sj += blocksize) {
                for (std::uint_fast32_t sk = 0; sk < Other; sk += blocksize) {
                    do_block(si, sj, sk, a, b, c);
                }
            }
        }
    }

private:
    static void
    do_block(const std::uint_fast32_t si, const std::uint_fast32_t sj, const std::uint_fast32_t sk, Type const* a,
             Type const* b, Type* c)
    {
        auto const last_si = si + blocksize;
        auto const last_sj = sj + blocksize;
        auto const last_sk = sk + blocksize;
        for (std::uint_fast32_t i = si; i < last_si; ++i) {
            for (std::uint_fast32_t j = sj; j < last_sj; ++j) {
                auto const current = i * Columns + j;
                Type       cij     = c[current];                  /* cij = C[i][j] */
                for (std::uint_fast32_t k = sk; k < last_sk; ++k) {
                    cij += a[i * Other + k] * b[k * Columns + j]; /* cij+=A[i][k]*B[k][j] */
                }
                c[current] = cij;                                 /* C[i][j] = cij */
            }
        }
    }
};

}    // namespace xitren::math

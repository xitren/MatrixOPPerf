#pragma once

#include <xitren/math/branchless.hpp>

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

// The `enum class optimization` defines an enumeration type that represents different optimization
// strategies for matrix multiplication algorithms. The enum class `optimization` includes the
// following options:
enum class optimization { naive, blocked, avx256, avx512, avx512_blocked, openmp_avx512_blocked };

// The `gemm_core` class template in the provided code is implementing a generic matrix
// multiplication (GEMM) algorithm for matrices of fixed size specified by the template parameters
// `Rows` and `Columns`. It supports different optimization strategies specified by the
// `optimization` enum class.
template <std::uint_fast32_t Rows, std::uint_fast32_t Columns, typename Type, optimization Alg>
class gemm_core {
    static_assert(Alg == optimization::naive, "Falling to base gemm!");

public:
    // The `mult` function in the provided code is performing matrix multiplication for matrices of
    // fixed size specified by the template parameters `Rows` and `Columns`.
    template <std::uint_fast32_t Other>
    static void
    mult(Type const* a, Type const* b, Type* c) noexcept
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

    // The `add` function is performing element-wise addition of two matrices represented by arrays
    // `a` and `b`, and storing the result in the array `c`.
    static void
    add(Type const* a, Type const* b, Type* c) noexcept
    {
        for (std::uint_fast32_t i = 0; i < Rows; ++i) {
            for (std::uint_fast32_t j = 0; j < Columns; ++j) {
                auto const current = i * Columns + j;
                Type       cij     = a[current] + b[current]; /* cij += A[i][j] + B[i][j] */
                c[current]         = cij;                     /* C[i][j] = cij */
            }
        }
    }

    // The `sub` function is performing element-wise subtraction of two matrices represented by
    // arrays `a` and `b`, and storing the result in the array `c`.
    static void
    sub(Type const* a, Type const* b, Type* c) noexcept
    {
        for (std::uint_fast32_t i = 0; i < Rows; ++i) {
            for (std::uint_fast32_t j = 0; j < Columns; ++j) {
                auto const current = i * Columns + j;
                Type       cij     = a[current] - b[current]; /* cij += A[i][j] - B[i][j] */
                c[current]         = cij;                     /* C[i][j] = cij */
            }
        }
    }

    // The `transpose` function is responsible for transposing a matrix represented by the input
    // array `a` and storing the transposed matrix in the output array `c`.
    static void
    transpose(Type const* a, Type* c) noexcept
    {
        for (std::uint_fast32_t i = 0; i < Rows; ++i) {
            for (std::uint_fast32_t j = 0; j < Columns; ++j) {
                Type& cij          = a[j * Columns + i]; /* cij += A[i][j] */
                c[i * Columns + j] = cij;                /* C[i][j] = cij */
            }
        }
    }

    // The `static Type trace(Type const* a) noexcept` function in the provided code is calculating
    // the trace of a square matrix represented by the input array `a`.
    static Type
    trace(Type const* a) noexcept
    {
        Type ret{};
        for (std::uint_fast32_t i = 0; i < Rows; ++i) {
            auto const current = i * Columns + i;
            Type&      cij     = a[current]; /* cij = A[i][i] */
            ret += cij;                      /* ret += cij */
        }
        return ret;
    }

    // The `static Type min(Type const* a) noexcept` function is calculating the minimum value
    // within a matrix represented by the input array `a`.
    static Type
    min(Type const* a) noexcept
    {
        Type ret{};
        for (std::uint_fast32_t i = 0; i < Rows; ++i) {
            for (std::uint_fast32_t j = 0; j < Columns; ++j) {
                auto const current = i * Columns + j;
                Type&      cij     = a[current]; /* cij = A[i][j] */
                ret                = branchless_select(cij < ret, cij, ret);
            }
        }
        return ret;
    }

    // The `static Type max(Type const* a) noexcept` function is calculating the maximum value
    // within a matrix represented by the input array `a`.
    static Type
    max(Type const* a) noexcept
    {
        Type ret{};
        for (std::uint_fast32_t i = 0; i < Rows; ++i) {
            for (std::uint_fast32_t j = 0; j < Columns; ++j) {
                auto const current = i * Columns + j;
                Type&      cij     = a[current]; /* cij = A[i][j] */
                ret                = branchless_select(cij > ret, cij, ret);
            }
        }
        return ret;
    }
};

// This code snippet is defining a specialization of the `gemm_core` class template for the
// optimization strategy `optimization::blocked`.
template <std::uint_fast32_t Rows, std::uint_fast32_t Columns, typename Type>
class gemm_core<Rows, Columns, Type, optimization::blocked>
    : gemm_core<Rows, Columns, Type, optimization::naive> {
    static constexpr std::uint_fast32_t blocksize = 32;
    static_assert(!(Rows % blocksize), "Should be dividable to blocksize!");
    static_assert(!(Columns % blocksize), "Should be dividable to blocksize!");

public:
    using gemm_core<Rows, Columns, Type, optimization::naive>::add;
    using gemm_core<Rows, Columns, Type, optimization::naive>::sub;
    using gemm_core<Rows, Columns, Type, optimization::naive>::transpose;
    using gemm_core<Rows, Columns, Type, optimization::naive>::trace;
    using gemm_core<Rows, Columns, Type, optimization::naive>::min;
    using gemm_core<Rows, Columns, Type, optimization::naive>::max;

    template <std::uint_fast32_t Other>
    static void
    mult(Type const* a, Type const* b, Type* c) noexcept
    {
        static_assert(!(Other % blocksize), "Should be dividable to blocksize!");
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
             Type const* a, Type const* b, Type* c) noexcept
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

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
#include <x86intrin.h>

namespace xitren::simd {


template <std::size_t Size>
void matrix_mult_basic(const std::array<double, Size>& A,
                       const std::array<double, Size>& B,
                       std::array<double, Size>& C) {
    constexpr uint32_t n = Size/2;
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            double cij = C[i + j * n]; /* cij = C[i][j] */
            for (uint32_t k = 0; k < n; k++) {
                cij += A[i + k * n] * B[k + j * n]; /* cij += A[i][k]*B[k][j] */
            }
            C[i + j * n] = cij; /* C[i][j] = cij */
        }
    }
}

/*
Assume matrix C is initialized to zero. The `do_block` function is basic
with new parameters to specify the starting positions of the submatrices of
BLOCKSIZE. The gcc optimizer can remove the function overhead instructions by
inlining the do_block function.

Instead of operating on entire rows or columns of an array, blocked algorithms
operate on submatrices or blocks. The goal is to maximize accesses to the data
loaded into the cache before the data are replaced; that is, improve temporal
locality to reduce cache misses.

It reads all N-by-N elements of B, reads the same N elements in what corresponds
to one row of A repeatedly, and writes what corresponds to one row of N elements
of C. If the cache can hold one N-by-N matrix and one row of N, then at least
the i-th row of A and the array B may stay in the cache. Less than that and
misses may occur for both B and C. In the worst case, there would be 2*N^3 + N^2
memory words accessed for N^3 operations.

To ensure that the elements being accessed can fit in the cache, the original
code is changed to compute on a submatrix. Hence, we essentially invoke the
version of repeatedly on matrices of size BLOCKSIZE by BLOCKSIZE.
BLOCKSIZE is called the blocking factor.

The function `do_block` with three new parameters si, sj, and sk to
specify the starting position of each submatrix of A, B, and C. The two inner
loops of the `do_block` now compute in steps of size BLOCKSIZE rather than the
full length of B and C . The gcc optimizer removes any function call overhead by
`inlining` the function; that is, it inserts the code directly to avoid the
conventional parameter passing and return address bookkeeping instructions.

Looking only at capacity misses, the total number of memory words accessed
is 2*N^3 / BLOCKSIZE + N^2 . This total is an improvement by about a factor of
BLOCKSIZE. Hence, blocking exploits a combination of spatial and temporal
locality, since A benefits from spatial locality and B benefits from temporal
locality. Depending on the computer and size of the matrices, blocking can
improve performance by about a factor of 2 to more than a factor of 10.

Although we have aimed at reducing cache misses, blocking can also be used to
help register allocation. By taking a small blocking size, such that the block
can be held in registers, we can minimize the number of loads and stores in the
program, which again improves performance.
*/

static constexpr uint32_t BLOCKSIZE = 32;

template <std::size_t Size>
static void do_block(const uint32_t n, const uint32_t si, const uint32_t sj,
                     const uint32_t sk, const std::array<double, Size>& A,
                     const std::array<double, Size>& B,
                     std::array<double, Size>& C) {
    static_assert((Size & (Size - 1)) == 0, "Should be power of 2!");
    for (uint32_t i = si; i < si + BLOCKSIZE; ++i) {
        for (uint32_t j = sj; j < sj + BLOCKSIZE; ++j) {
            double cij = C[i + j * n]; /* cij = C[i][j] */
            for (uint32_t k = sk; k < sk + BLOCKSIZE; k++) {
                cij += A[i + k * n] * B[k + j * n]; /* cij+=A[i][k]*B[k][j] */
            }
            C[i + j * n] = cij; /* C[i][j] = cij */
        }
    }
}

template <std::size_t Size>
void matrix_mult_basic_blocked(const std::array<double, Size>& A,
                               const std::array<double, Size>& B,
                               std::array<double, Size>& C) {
    static_assert((Size & (Size - 1)) == 0, "Should be power of 2!");
    constexpr uint32_t n = Size/2;
    for (uint32_t sj = 0; sj < n; sj += BLOCKSIZE) {
        for (uint32_t si = 0; si < n; si += BLOCKSIZE) {
            for (uint32_t sk = 0; sk < n; sk += BLOCKSIZE) {
                do_block(n, si, sj, sk, A, B, C);
            }
        }
    }
}


/*
The declaration uses the __m256d datatype, which tells the compiler the variable
will hold four double-precision floating-point values. The intrinsic
_mm256_load_pd() uses AVX instructions to load four double-precision
floating-point numbers in parallel (_pd) from the matrix C into c0. The address
calculation C+i+j*n represents elementC[i+j*n]. Symmetrically, the final step
uses the intrinsic_mm256_store_pd() to store four double-precision
floating-pointnumbers from c0 into the matrix C. As we’re going through four
elements each iteration, the outer for loop increments i by 4 instead of by 1.

Inside the loops, we first load four elements of A againusing _mm256_load_pd().
To multiply these elements by one elementof B, we first use the intrinsic
_mm256_broadcast_sd(), which makes four identical copies of the scalar double
precision number - in this case an element of B - in one of the YMM registers.
We then use _mm256_mul_pd() to multiply the four double-precision results in
parallel. Finally, _mm256_add_pd() adds the four products to the four sums in
c0.

For matrices of dimensions of 32 by 32, the unoptimized DGEMM (function called
`basic`) runs at 1.7 GigaFLOPS (FLoating point OperationsPer Second) on one core
of a 2.6 GHz Intel Core i7 (Sandy Bridge). The optimized code (function called
`avx256`) performs at 6.4 GigaFLOPS. The AVX-256 version is 3.85 times as fast,
which is very close to the factor of 4.0 increase that you might hope for from
performing four times as many operations at a time by using subword parallelism.
*/
template <std::size_t Size>
void matrix_mult_avx256(const std::array<double, Size>& A,
                        const std::array<double, Size>& B,
                        std::array<double, Size>& C) {
    static_assert((Size & (Size - 1)) == 0, "Should be power of 2!");
    constexpr uint32_t n = Size/2;
    for (uint32_t i = 0; i < n; i += 4) {
        for (uint32_t j = 0; j < n; j++) {
            __m256d c0 = _mm256_load_pd(C + i + j * n); /* c0 = C[i][j] */
            for (uint32_t k = 0; k < n; k++) {
                c0 = _mm256_add_pd(
                    c0, /* c0 += A[i][k]*B[k][j] */
                    _mm256_mul_pd(_mm256_load_pd(A + i + k * n),
                                  _mm256_broadcast_sd(B + k + j * n)));
            }

            _mm256_store_pd(C + i + j * n, c0); /* C[i][j] = c0 */
        }
    }
}

/*
Optimized version of DGEMM using C intrinsics to generate the AVX512.

To demonstrate the performance impact of subwordparallelism, we rerun the code
using AVX. While compiler writers may eventually be able to routinely produce
high-quality code that uses the AVX instructions of the x86, for now we must
`cheat` by using C intrinsics that more or less tell the compiler exactly how to
produce good code.

Th declaration uses the __m512d data type, which tells the compiler the
variable will hold 8 double-precision floating-point values (8 × 64 bits =
512 bits). The intrinsic _mm512_load_pd() uses AVX
instructions to load 8 double-precision floating-point numbers in parallel ( _pd
) from the matrix C into c0. The address calculation C+i+j*n represents element
C[i+j*n]. Symmetrically, the final step uses the intrinsic _mm512_store_pd() to
store 8 double-precision floating-point numbers from c0 into the matrix C. As we
are going through 8 elements each iteration, the outer for loop increments `i`
by 8.

Inside the loops, we first load 8 elements of A again using
_mm512_ load_pd(). To multiply these elements by one element of B, we
first use the intrinsic _mm512_broadcast_sd(), which makes eight identical
copies of the scalar double precision number (in this case an element of B)
in one of the ZMM registers. We then use _mm512_fmadd_pd to
multiply the 8 double-precision results in parallel and then add the 8
products to the 8 sums in c0.

*/
template <std::size_t Size>
void matrix_mult_avx512(const std::array<double, Size>& A,
                        const std::array<double, Size>& B,
                        std::array<double, Size>& C) {
    static_assert((Size & (Size - 1)) == 0, "Should be power of 2!");
    constexpr uint32_t n = Size/2;
    for (uint32_t i = 0; i < n; i += 8) {
        for (uint32_t j = 0; j < n; ++j) {
            __m512d c0 = _mm512_load_pd(C + i + j * n);  // c0 = C[i][j]
            for (uint32_t k = 0; k < n; k++) {
                // c0 += A[i][k] * B[k][j]
                __m512d bb = _mm512_broadcastsd_pd(_mm_load_sd(B + j * n + k));
                c0 = _mm512_fmadd_pd(_mm512_load_pd(A + n * k + i), bb, c0);
            }
            _mm512_store_pd(C + i + j * n, c0);  // C[i][j] = c0
        }
    }
}

/*
Optimized version of DGEMM using C intrinsics to generate the AVX
subword-parallel instructions for the x86 and loop unrolling to create more
opportunities for instruction-level parallelism.

We can see the impact of
instruction-level parallelism by unrolling the loop so that the multiple-issue,
out-of-order execution processor has more instructions to work with.
The function below is the unrolled version of function avx512, which contains
the C intrinsics to produce the AVX-512 instructions.
*/
template <std::size_t Size>
void matrix_mult_unrolled(const std::array<double, Size>& A,
                          const std::array<double, Size>& B,
                          std::array<double, Size>& C) {
    static_assert((Size & (Size - 1)) == 0, "Should be power of 2!");
    constexpr uint32_t UNROLL = 4;
    constexpr uint32_t n = Size/2;

    for (uint32_t i = 0; i < n; i += UNROLL * 8) {
        for (uint32_t j = 0; j < n; ++j) {
            __m512d c[UNROLL];
            for (uint32_t r = 0; r < UNROLL; r++) {
                c[r] = _mm512_load_pd(C.data() + i + r * 8 + j * n);  //[ UNROLL];
            }

            for (uint32_t k = 0; k < n; k++) {
                __m512d bb = _mm512_broadcastsd_pd(_mm_load_sd(B.data() + j * n + k));
                for (uint32_t r = 0; r < UNROLL; r++) {
                    c[r] = _mm512_fmadd_pd(
                        _mm512_load_pd(A.data() + n * k + r * 8 + i), bb, c[r]);
                }
            }
            for (uint32_t r = 0; r < UNROLL; r++) {
                _mm512_store_pd(C.data() + i + r * 8 + j * n, c[r]);
            }
        }
    }
}

/*
Optimized version of DGEMM using C intrinsics to generate the AVX
subword-parallel instructions for the x86, loop unrolling and blocking to create
more opportunities for instruction-level parallelism.
*/

template <std::size_t Size>
static void do_block_simd(const uint32_t n, const uint32_t si, const uint32_t sj,
                     const uint32_t sk, const std::array<double, Size>& A,
                     const std::array<double, Size>& B,
                     std::array<double, Size>& C) {
    static_assert((Size & (Size - 1)) == 0, "Should be power of 2!");
    constexpr uint32_t UNROLL = 4;

    for (uint32_t i = si; i < si + BLOCKSIZE; i += UNROLL * 8) {
        for (uint32_t j = sj; j < sj + BLOCKSIZE; ++j) {
            __m512d c[UNROLL];
            for (uint32_t r = 0; r < UNROLL; r++) {
                c[r] = _mm512_load_pd(C.data() + i + r * 8 + j * n);  //[ UNROLL];
            }

            for (uint32_t k = sk; k < sk + BLOCKSIZE; k++) {
                __m512d bb = _mm512_broadcastsd_pd(_mm_load_sd(B.data() + j * n + k));
                for (uint32_t r = 0; r < UNROLL; r++) {
                    c[r] = _mm512_fmadd_pd(
                        _mm512_load_pd(A.data() + n * k + r * 8 + i), bb, c[r]);
                }
            }
            for (uint32_t r = 0; r < UNROLL; r++) {
                _mm512_store_pd(C.data() + i + r * 8 + j * n, c[r]);
            }
        }
    }
}

template <std::size_t Size>
void matrix_mult_openmp(const std::array<double, Size>& A,
                        const std::array<double, Size>& B,
                        std::array<double, Size>& C) {
    static_assert((Size & (Size - 1)) == 0, "Should be power of 2!");
    constexpr uint32_t n = Size/2;
#pragma omp parallel for
    for (uint32_t sj = 0; sj < n; sj += BLOCKSIZE) {
        for (uint32_t si = 0; si < n; si += BLOCKSIZE) {
            for (uint32_t sk = 0; sk < n; sk += BLOCKSIZE) {
                do_block_simd(n, si, sj, sk, A, B, C);
            }
        }
    }
}

}  // namespace xitren::simd

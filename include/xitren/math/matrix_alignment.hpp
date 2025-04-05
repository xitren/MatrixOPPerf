#pragma once

#include <xitren/math/gemm_core.hpp>

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

template <class Type, std::size_t Rows, std::size_t Columns, xitren::math::optimization Alg>
class matrix_aligned {
    using Core = xitren::math::gemm_core<Rows, Columns, Type, Alg>;

    static_assert(noexcept(Core::template mult<32>(nullptr, nullptr, nullptr)));
    static_assert(noexcept(Core::add(nullptr, nullptr, nullptr)));
    static_assert(noexcept(Core::sub(nullptr, nullptr, nullptr)));
    static_assert(noexcept(Core::transpose(nullptr, nullptr)));
    static_assert(noexcept(Core::trace(nullptr)));
    static_assert(noexcept(Core::min(nullptr)));
    static_assert(noexcept(Core::max(nullptr)));

public:
    matrix_aligned()
    {
        data_ = (Type*)_mm_malloc(Rows * Columns * sizeof(Type), 64);
        if (data_ == nullptr)
            throw std::bad_alloc{};    // ("failed to allocate largest problem size");
    }
    ~matrix_aligned() { _mm_free(data_); }

    template <std::size_t ColumnsOther>
    static void
    mult(matrix_aligned<Type, Rows, ColumnsOther, Alg> const&    a,
         matrix_aligned<Type, ColumnsOther, Columns, Alg> const& b,
         matrix_aligned<Type, Rows, Columns, Alg>&               c)
    {
        Core::template mult<ColumnsOther>(a.data_, b.data_, c.data_);
    }

    auto&
    get(std::size_t row, std::size_t column)
    {
        return data_[(row * Columns) + column];
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

    Type* data_{nullptr};

    static std::shared_ptr<matrix_aligned>
    get_zeros_matrix()
    {
        std::shared_ptr<matrix_aligned> ret = std::make_shared<matrix_aligned>();
        for (std::uint32_t i = 0; i < Rows * Columns; ++i) {
            ret->data_[i] = 0;
        }
        return ret;
    }

private:
};

}    // namespace xitren::math

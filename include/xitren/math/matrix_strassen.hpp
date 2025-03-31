#pragma once

#include <xitren/math/branchless.hpp>

#include <x86intrin.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <vector>

namespace xitren::math {

template <class T>
struct custom_allocator {
    static constexpr std::size_t align = 64;

    using value_type            = T;
    custom_allocator() noexcept = default;

    template <class U>
    custom_allocator(custom_allocator<U> const&) noexcept
    {}

    T*
    allocate(std::size_t n)
    {
        return (T*)_mm_malloc(n * sizeof(T), align);
    }

    void
    deallocate(T* p, std::size_t n)
    {
        _mm_free(p);
    }
};

template <class Type, std::size_t Size>
class matrix_strassen {
    static_assert((Size & (Size - 1)) == 0, "Should be power of 2!");

    static constexpr std::size_t capacity         = Size * Size;
    static constexpr std::size_t quarter_capacity = Size * Size / 4;
    static constexpr std::size_t align            = 64;

public:
    using quarter_type      = matrix_strassen<Type, Size / 2>;
    using data_type         = Type*;
    using array_type        = std::array<Type, capacity>;
    using quarter_data_type = std::array<Type, quarter_capacity>;

    matrix_strassen() = default;
    matrix_strassen(array_type const& data)
        : a_{get_init(data, 0)}, b_{get_init(data, 1)}, c_{get_init(data, 2)}, d_{get_init(data, 3)}
    {}

    inline auto&
    get(std::size_t row, std::size_t column)
    {
        auto const half_size = Size >> 1;
        auto&      sel1      = branchless_select(column < half_size, a_, b_);
        auto&      sel2      = branchless_select(column < half_size, c_, d_);
        auto&      sel_f     = branchless_select(row < half_size, sel1, sel2);
        return sel_f.get(row % half_size, column % half_size);
    }

    inline void
    mult(matrix_strassen const& other, matrix_strassen& ret)
    {
        auto const H1 = (a_ + d_) * (other.a_ + other.d_);
        auto const H2 = (c_ + d_) * other.a_;
        auto const H3 = a_ * (other.b_ - other.d_);
        auto const H4 = d_ * (other.c_ - other.a_);
        auto const H5 = (a_ + b_) * other.d_;
        auto const H6 = (c_ - a_) * (other.a_ + other.b_);
        auto const H7 = (b_ - d_) * (other.c_ + other.d_);
        ret.a_        = H1 + H4 - H5 + H7;
        ret.b_        = H3 + H5;
        ret.c_        = H2 + H4;
        ret.d_        = H1 + H3 - H2 + H6;
    }

    inline void
    add(matrix_strassen const& other, matrix_strassen& ret)
    {
        ret.a_ = a_ + other.a_;
        ret.b_ = b_ + other.b_;
        ret.c_ = c_ + other.c_;
        ret.d_ = d_ + other.d_;
    }

    inline void
    sub(matrix_strassen const& other, matrix_strassen& ret)
    {
        ret.a_ = a_ - other.a_;
        ret.b_ = b_ - other.b_;
        ret.c_ = c_ - other.c_;
        ret.d_ = d_ - other.d_;
    }

    inline matrix_strassen
    operator*(matrix_strassen const& other) const
    {
        if constexpr (Size >= 64) {
            static constexpr std::size_t                     ops = 7;
            std::array<std::function<quarter_type(void)>, 7> funcs{
                {{[&]() -> quarter_type { return (a_ + d_) * (other.a_ + other.d_); }},
                 {[&]() -> quarter_type { return (c_ + d_) * other.a_; }},
                 {[&]() -> quarter_type { return a_ * (other.b_ - other.d_); }},
                 {[&]() -> quarter_type { return d_ * (other.c_ - other.a_); }},
                 {[&]() -> quarter_type { return (a_ + b_) * other.d_; }},
                 {[&]() -> quarter_type { return (c_ - a_) * (other.a_ + other.b_); }},
                 {[&]() -> quarter_type { return (b_ - d_) * (other.c_ + other.d_); }}}};
            std::array<quarter_type, ops> H;

#pragma omp parallel for
            for (std::size_t i = 0; i < H.size(); i++) {
                H[i] = (funcs[i])();
            }

            auto const Q1 = H[0] + H[3] - H[4] + H[6];
            auto const Q2 = H[2] + H[4];
            auto const Q3 = H[1] + H[3];
            auto const Q4 = H[0] + H[2] - H[1] + H[5];
            return matrix_strassen{Q1, Q2, Q3, Q4};
        } else {
            auto const H1 = (a_ + d_) * (other.a_ + other.d_);
            auto const H2 = (c_ + d_) * other.a_;
            auto const H3 = a_ * (other.b_ - other.d_);
            auto const H4 = d_ * (other.c_ - other.a_);
            auto const H5 = (a_ + b_) * other.d_;
            auto const H6 = (c_ - a_) * (other.a_ + other.b_);
            auto const H7 = (b_ - d_) * (other.c_ + other.d_);
            auto const Q1 = H1 + H4 - H5 + H7;
            auto const Q2 = H3 + H5;
            auto const Q3 = H2 + H4;
            auto const Q4 = H1 + H3 - H2 + H6;
            return matrix_strassen{Q1, Q2, Q3, Q4};
        }
    }

    inline matrix_strassen
    operator+(matrix_strassen const& other) const
    {
        return matrix_strassen{a_ + other.a_, b_ + other.b_, c_ + other.c_, d_ + other.d_};
    }

    matrix_strassen inline
    operator-(matrix_strassen const& other) const
    {
        return matrix_strassen{a_ + other.a_, b_ + other.b_, c_ + other.c_, d_ + other.d_};
    }

    void
    clear()
    {
        a_.clear();
        b_.clear();
        c_.clear();
        d_.clear();
    }

    static matrix_strassen
    get_rand_matrix()
    {
        auto a = quarter_type::get_rand_matrix();
        auto b = quarter_type::get_rand_matrix();
        auto c = quarter_type::get_rand_matrix();
        auto d = quarter_type::get_rand_matrix();
        return matrix_strassen{a, b, c, d};
    }

private:
    quarter_type a_{};
    quarter_type b_{};
    quarter_type c_{};
    quarter_type d_{};

    matrix_strassen(quarter_type const& m_a, quarter_type const& m_b, quarter_type const& m_c, quarter_type const& m_d)
        : a_{std::move(m_a)}, b_{std::move(m_b)}, c_{std::move(m_c)}, d_{std::move(m_d)}
    {}

    static constexpr quarter_data_type
    get_init(array_type const& data, std::size_t k)
    {
        quarter_data_type ret;
        std::size_t       x{};
        std::size_t       y{};
        std::size_t       j{};
        switch (k) {
        case 1:
            x = Size / 2;
            break;
        case 2:
            y = Size / 2;
            break;
        case 3:
            x = Size / 2;
            y = Size / 2;
            break;
        default:
            break;
        }
        for (std::size_t l{}; l < (Size / 2); l++) {
            for (std::size_t m{}; m < (Size / 2); m++) {
                ret[j++] = data[(l + y) * Size + m + x];
            }
        }
        return ret;
    }
};

template <class Type>
class matrix_strassen<Type, 2> {

    static constexpr std::size_t size     = 2;
    static constexpr std::size_t capacity = size * size;
    static constexpr std::size_t align    = 64;

public:
    using data_type  = std::vector<Type, custom_allocator<Type>>;
    using array_type = std::array<Type, capacity>;

    matrix_strassen() { data_.reserve(4); }
    matrix_strassen(array_type const& data)
    {
        data_.reserve(4);
        std::copy(data.begin(), data.end(), data_.begin());
    }
    matrix_strassen&
    operator=(matrix_strassen const& other)
    {
        data_.reserve(4);
        std::copy(other.data_.begin(), other.data_.end(), data_.begin());
        return *this;
    }
    matrix_strassen&
    operator=(matrix_strassen&& other)
    {
        data_.reserve(4);
        std::copy(other.data_.begin(), other.data_.end(), data_.begin());
        return *this;
    }
    matrix_strassen(matrix_strassen const& val)
    {
        data_.reserve(4);
        std::copy(val.data_.begin(), val.data_.end(), data_.begin());
    }
    matrix_strassen(matrix_strassen&& val) : data_{val.data_} {}
    ~matrix_strassen() {}

    auto&
    get(std::size_t row, std::size_t column)
    {
        return data_[(row << 1) + column];
    }

    void
    clear()
    {
        for (std::size_t i{}; i < capacity; i++) {
            data_[i] = 0;
        }
    }

    inline void
    mult(matrix_strassen const& other, matrix_strassen& ret)
    {
        Type const& a = data_[0];
        Type const& b = data_[1];
        Type const& c = data_[2];
        Type const& d = data_[3];
        Type const& A = other.data_[0];    // NOLINT
        Type const& C = other.data_[1];    // NOLINT
        Type const& B = other.data_[2];    // NOLINT
        Type const& D = other.data_[3];    // NOLINT

        const Type t = a * A;
        const Type u = (c - a) * (C - D);
        const Type v = (c + d) * (C - A);
        const Type w = t + (c + d - a) * (A + D - C);

        ret[0] = t + b * B;
        ret[1] = w + v + (a + b - c - d) * D;
        ret[2] = w + u + d * (B + C - A - D);
        ret[3] = w + u + v;
    }

    inline void
    add(matrix_strassen const& other, matrix_strassen& ret)
    {
        for (int i{}; i < capacity; i++) {
            ret[i] = data_[i] + other.data_[i];
        }
    }

    inline void
    sub(matrix_strassen const& other, matrix_strassen& ret)
    {
        for (int i{}; i < capacity; i++) {
            ret[i] = data_[i] - other.data_[i];
        }
    }

    inline matrix_strassen
    operator*(matrix_strassen const& other) const
    {
        Type const& a = data_[0];
        Type const& b = data_[1];
        Type const& c = data_[2];
        Type const& d = data_[3];
        Type const& A = other.data_[0];    // NOLINT
        Type const& C = other.data_[1];    // NOLINT
        Type const& B = other.data_[2];    // NOLINT
        Type const& D = other.data_[3];    // NOLINT

        const Type t = a * A;
        const Type u = (c - a) * (C - D);
        const Type v = (c + d) * (C - A);
        const Type w = t + (c + d - a) * (A + D - C);

        return matrix_strassen{t + b * B, w + v + (a + b - c - d) * D, w + u + d * (B + C - A - D), w + u + v};
    }

    inline matrix_strassen
    operator+(matrix_strassen const& other) const
    {
        return matrix_strassen{data_[0] + other.data_[0], data_[1] + other.data_[1], data_[2] + other.data_[2],
                               data_[3] + other.data_[3]};
    }

    inline matrix_strassen
    operator-(matrix_strassen const& other) const
    {
        return matrix_strassen{data_[0] - other.data_[0], data_[1] - other.data_[1], data_[2] - other.data_[2],
                               data_[3] - other.data_[3]};
    }

    static matrix_strassen
    get_rand_matrix()
    {
        std::srand(std::time({}));    // use current time as seed for random generator
        std::array<Type, size * size> data_rand;
        for (auto it{data_rand.begin()}; it != data_rand.end(); it++) {
            (*it) = static_cast<Type>(std::rand());
        }
        return matrix_strassen{data_rand};
    }

private:
    data_type data_{};

    matrix_strassen(Type const& m_a, Type const& m_b, Type const& m_c, Type const& m_d)
    {
        data_.reserve(4);
        data_[0] = m_a;
        data_[1] = m_b;
        data_[2] = m_c;
        data_[3] = m_d;
    }
};

}    // namespace xitren::math

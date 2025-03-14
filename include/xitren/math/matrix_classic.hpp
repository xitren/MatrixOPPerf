#pragma once

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
class matrix_classic : public std::array<std::array<Type, Columns>, Rows> {

public:
    using data_type = std::array<std::array<Type, Columns>, Rows>;

    matrix_classic() = default;
    matrix_classic(data_type const& data) : data_type{data} {}

    template <std::size_t ColumnsOther>
    matrix_classic<Type, Rows, ColumnsOther>
    operator*(matrix_classic<Type, Columns, ColumnsOther> const& other) const
    {
        typename matrix_classic<Type, Rows, ColumnsOther>::data_type ret;
        for (std::size_t i = 0; i < Rows; i++) {
            for (std::size_t j = 0; j < ColumnsOther; j++) {
                for (std::size_t k = 0; k < Columns; k++) {
                    ret[i][j] += data_type::operator[](i)[k] * other[k][j];
                }
            }
        }
        return matrix_classic<Type, Rows, ColumnsOther>{ret};
    }

    matrix_classic
    operator+(matrix_classic const& other) const
    {
        data_type ret;
        for (std::size_t i = 0; i < Rows; i++) {
            for (std::size_t j = 0; j < Columns; j++) {
                ret[i][j] = data_type::operator[](i)[j] + other[i][j];
            }
        }
        return matrix_classic{ret};
    }

    matrix_classic
    operator-(matrix_classic const& other) const
    {
        data_type ret;
        for (std::size_t i = 0; i < Rows; i++) {
            for (std::size_t j = 0; j < Columns; j++) {
                ret[i][j] = data_type::operator[](i)[j] - other[i][j];
            }
        }
        return matrix_classic{ret};
    }

    template <std::size_t ColumnsOther>
    void
    mult(matrix_classic<Type, Columns, ColumnsOther> const& other, matrix_classic<Type, Rows, ColumnsOther>& ret)
    {
        for (std::size_t i = 0; i < Rows; i++) {
            for (std::size_t j = 0; j < ColumnsOther; j++) {
                for (std::size_t k = 0; k < Columns; k++) {
                    ret[i][j] += data_type::operator[](i)[k] * other[k][j];
                }
            }
        }
    }

    void
    add(matrix_classic const& other, matrix_classic& ret)
    {
        for (std::size_t i = 0; i < Rows; i++) {
            for (std::size_t j = 0; j < Columns; j++) {
                ret[i][j] = data_type::operator[](i)[j] + other[i][j];
            }
        }
    }

    void
    sub(matrix_classic const& other, matrix_classic& ret)
    {
        for (std::size_t i = 0; i < Rows; i++) {
            for (std::size_t j = 0; j < Columns; j++) {
                ret[i][j] = data_type::operator[](i)[j] - other[i][j];
            }
        }
    }

    static std::shared_ptr<matrix_classic>
    get_rand_matrix(double max_val, double min_val)
    {
        std::shared_ptr<matrix_classic>  ret = std::make_shared<matrix_classic>();
        std::random_device               rd;
        std::mt19937                     gen(rd());
        std::uniform_real_distribution<> dis(min_val, max_val);
        std::srand(std::time({}));    // use current time as seed for random generator
        for (auto it{ret->begin()}; it != ret->end(); it++) {
            for (auto it2{(*it).begin()}; it2 != ((*it).end()); it2++) {
                (*it2) = static_cast<Type>(dis(gen));
            }
        }
        return ret;
    }
};

}    // namespace xitren::math

#pragma once

#include <xitren/math/matrix_strassen.hpp>

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

template <class Type, std::size_t Rows, std::size_t Columns, std::size_t Batch = 0>
class matrix {

    static_assert(Rows > 1 && Columns > 1);

    static constexpr std::size_t
    get_max_item()
    {
        if (Batch != 0) {
            return Batch;
        }
        std::array<std::size_t, 7> sizes{{128, 64, 32, 16, 8, 4, 2}};

        std::size_t max_row{2};
        for (auto& size : sizes) {
            if ((Rows / size) != 0) {
                max_row = size;
            }
            break;
        }
        std::size_t max_col{2};
        for (auto& size : sizes) {
            if ((Columns / size) != 0) {
                max_col = size;
            }
            break;
        }
        return std::min(max_row, max_col);
    }

public:
    static constexpr auto batch_value = get_max_item();

    using init_type        = std::array<Type, Rows * Columns>;
    using batch_type       = matrix_strassen<Type, batch_value>;
    using batch_store_type = std::shared_ptr<batch_type>;

    static constexpr auto batch_rows    = Rows / batch_value;
    static constexpr auto batch_columns = Columns / batch_value;
    static constexpr auto rest_rows     = Rows % batch_value;
    static constexpr auto rest_columns  = Columns % batch_value;

    static constexpr auto batch_rows_end    = (batch_rows)*batch_value;
    static constexpr auto batch_columns_end = (batch_columns)*batch_value;

    using data_type         = std::array<std::array<batch_type, batch_columns>, Rows / batch_value>;
    using rest_rows_type    = std::array<std::array<Type, Columns>, rest_rows>;
    using rest_columns_type = std::array<std::array<Type, rest_columns>, Rows - rest_rows>;

    matrix() = default;
    matrix(init_type const& data)
    {
        // ToDo: Rewrite this to init version
        for (std::size_t row{}; row < Rows; row++) {
            for (std::size_t column{}; column < Columns; column++) {
                get(row, column) = data[row * Columns + column];
            }
        }
    }

    Type&
    get(std::size_t row, std::size_t column)
    {
        if ((row < batch_rows_end) && (column < batch_columns_end)) {
            auto const i   = row / batch_value;
            auto const j   = column / batch_value;
            auto&      val = batched_section[i][j].get(row % batch_value, column % batch_value);
            return val;
        } else {
            if ((row < batch_rows_end) && (column >= batch_columns_end)) {
                return rest_columns_section[row][column - batch_columns_end];
            } else {
                auto xr = row - batch_rows_end;
                auto xc = column - batch_columns_end;
                return rest_rows_section[row - batch_rows_end][column];
            }
        }
    }

    template <class T, std::size_t R, std::size_t C, std::size_t B>
    friend class matrix;

    //
    // Hybrid mult                                                             | |B B| C |
    //                                                                         | |B B| C |
    //                                          other[Columns][ColumnsOther] = | |B B| C |
    //                                                                         | |B B| C |
    //                                                                         |  R R  R |
    //
    //
    //                           | |B B| |B B| C |                             | |B B| C |
    //  (*this)[Rows][Columns] = | |B B| |B B| C |   ret[Rows][ColumnsOther] = | |B B| C |
    //                           |  R R   R R  R |                             |  R R  R |
    template <std::size_t ColumnsOther>
    void
    mult(matrix<Type, Columns, ColumnsOther, Batch>& other, matrix<Type, Rows, ColumnsOther, Batch>& ret)
    {
        // ToDo: get back to const
        static_assert(batch_value == other.batch_value);
        using ret_type = matrix<Type, Rows, ColumnsOther, batch_value>;
        // Calculate batch zone
        for (std::size_t i = 0; i < ret_type::batch_rows; i++) {
            for (std::size_t j = 0; j < ret_type::batch_columns; j++) {
                ret.batched_section[i][j].clear();
                for (std::size_t k = 0; k < batch_columns; k++) {
                    ret.batched_section[i][j]
                        = ret.batched_section[i][j] + batched_section[i][k] * other.batched_section[k][j];
                }
                batch_type rest{};
                auto const ix = i * batch_value;
                auto const jy = j * batch_value;
                for (std::size_t x = 0; x < batch_value; x++) {
                    for (std::size_t y = 0; y < batch_value; y++) {
                        auto& rest_item = rest.get(x, y);
                        rest_item       = 0;
                        for (std::size_t z = 0; z < rest_rows; z++) {
                            rest_item += rest_columns_section[x + ix][z] * other.rest_rows_section[z][y + jy];
                        }
                    }
                }
                ret.batched_section[i][j] = ret.batched_section[i][j] + rest;
            }
        }
        // Calculate rest columns zone
        for (std::size_t i = 0; i < ret.rest_columns_section.size(); i++) {
            for (std::size_t j = 0; j < ret_type::rest_columns; j++) {
                ret.rest_columns_section[i][j] = 0;
                auto const m_col               = j + ret_type::batch_columns_end;
                for (std::size_t k = 0; k < Columns; k++) {
                    ret.rest_columns_section[i][j] += get(i, k) * other.get(k, m_col);
                }
            }
        }
        // Calculate rest rows zone
        for (std::size_t i = 0; i < rest_rows; i++) {
            for (std::size_t j = 0; j < Columns; j++) {
                ret.rest_rows_section[i][j] = 0;
                auto const m_row            = i + batch_rows_end;
                for (std::size_t k = 0; k < Columns; k++) {
                    ret.rest_rows_section[i][j] += get(m_row, k) * other.get(k, j);
                }
            }
        }
    }

    void
    add(matrix const& other, matrix& ret) const
    {
        static_assert(batch_value == other.batch_value);
        for (std::size_t i = 0; i < batch_rows; i++) {
            for (std::size_t j = 0; j < batch_columns; j++) {
                ret.batched_section[i][j] = batched_section[i][j] + other.batched_section[i][j];
            }
        }
        for (std::size_t i = 0; i < rest_columns_section.size(); i++) {
            for (std::size_t j = 0; j < rest_columns; j++) {
                ret.rest_columns_section[i][j] = rest_columns_section[i][j] + other.rest_columns_section[i][j];
            }
        }
        for (std::size_t i = 0; i < rest_rows; i++) {
            for (std::size_t j = 0; j < Columns; j++) {
                ret.rest_rows_section[i][j] = rest_rows_section[i][j] + other.rest_rows_section[i][j];
            }
        }
    }

    void
    sub(matrix const& other, matrix& ret) const
    {
        static_assert(batch_value == other.batch_value);
        for (std::size_t i = 0; i < batch_rows; i++) {
            for (std::size_t j = 0; j < batch_columns; j++) {
                ret.batched_section[i][j] = batched_section[i][j] - other.batched_section[i][j];
            }
        }
        for (std::size_t i = 0; i < rest_columns_section.size(); i++) {
            for (std::size_t j = 0; j < rest_columns; j++) {
                ret.rest_columns_section[i][j] = rest_columns_section[i][j] - other.rest_columns_section[i][j];
            }
        }
        for (std::size_t i = 0; i < rest_rows; i++) {
            for (std::size_t j = 0; j < Columns; j++) {
                ret.rest_rows_section[i][j] = rest_rows_section[i][j] - other.rest_rows_section[i][j];
            }
        }
    }

    static matrix
    get_rand_matrix()
    {
        matrix ret{};
        std::srand(std::time({}));    // use current time as seed for random generator
        for (std::size_t i = 0; i < Rows; i++) {
            for (std::size_t j = 0; j < Columns; j++) {
                ret.get(i, j) = static_cast<Type>(std::rand());
            }
        }
        return ret;
    }

private:
    data_type         batched_section;
    rest_columns_type rest_columns_section;
    rest_rows_type    rest_rows_section;
};

}    // namespace xitren::math

#pragma once

#include <algorithm>
#include <cstdint>

namespace xitren::math {

namespace detail {
template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector {
    using value_t = std::false_type;
    using type    = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
    using type    = Op<Args...>;
};

}    // namespace detail

// special type to indicate detection failure
struct nonesuch {
    nonesuch()                = delete;
    ~nonesuch()               = delete;
    nonesuch(nonesuch const&) = delete;
    void
    operator=(nonesuch const&)
        = delete;
};

template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<nonesuch, void, Op, Args...>::value_t;

template <template <class...> class Op, class... Args>
using detected_t = typename detail::detector<nonesuch, void, Op, Args...>::type;

template <class Default, template <class...> class Op, class... Args>
using detected_or = detail::detector<Default, void, Op, Args...>;

}    // namespace xitren::math

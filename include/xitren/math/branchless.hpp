#pragma once

#include <algorithm>
#include <cstdint>

namespace xitren::math {

// The `branchless_select` function is a branchless way to select between two values `a` and `b`
// based on the value of the `compare` parameter.Instead of using traditionalconditional
// statements(if - else), it uses bitwise operations to perform the selection without branching.
template <class Type>
static inline Type&
branchless_select(int compare, Type& a, Type& b)
{
    auto  ptr_a = reinterpret_cast<std::uintptr_t>(&a);
    auto  ptr_b = reinterpret_cast<std::uintptr_t>(&b);
    auto  ptr_c = ((compare - 1) & (ptr_b ^ ptr_a)) ^ ptr_a;
    Type& c     = *(reinterpret_cast<Type*>(ptr_c));
    return c;
}
}    // namespace xitren::math
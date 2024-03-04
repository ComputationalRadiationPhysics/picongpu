/* Copyright 2022 Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/meta/Integral.hpp"

#include <algorithm>
#include <limits>

namespace alpaka::core
{
    //! \return The input casted and clipped to T.
    template<typename T, typename V>
    auto clipCast(V const& val) -> T
    {
        static_assert(
            std::is_integral_v<T> && std::is_integral_v<V>,
            "clipCast can not be called with non-integral types!");

        constexpr auto max = static_cast<V>(std::numeric_limits<alpaka::meta::LowerMax<T, V>>::max());
        constexpr auto min = static_cast<V>(std::numeric_limits<alpaka::meta::HigherMin<T, V>>::min());

        return static_cast<T>(std::max(min, std::min(max, val)));
    }
} // namespace alpaka::core

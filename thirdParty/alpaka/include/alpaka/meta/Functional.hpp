/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

namespace alpaka::meta
{
    template<typename T>
    struct min
    {
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC constexpr auto operator()(T const& lhs, T const& rhs) const
        {
            return (lhs < rhs) ? lhs : rhs;
        }
    };

    template<typename T>
    struct max
    {
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC constexpr auto operator()(T const& lhs, T const& rhs) const
        {
            return (lhs > rhs) ? lhs : rhs;
        }
    };
} // namespace alpaka::meta

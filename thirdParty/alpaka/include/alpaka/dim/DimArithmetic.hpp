/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dim/DimIntegralConst.hpp"

#include <type_traits>

namespace alpaka::trait
{
    //! The arithmetic type dimension getter trait specialization.
    template<typename T>
    struct DimType<T, std::enable_if_t<std::is_arithmetic_v<T>>>
    {
        using type = DimInt<1u>;
    };
} // namespace alpaka::trait

/* Copyright 2020 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dim/Traits.hpp"

#include <type_traits>

namespace alpaka
{
    // N(th) dimension(s).
    template<std::size_t N>
    using DimInt = std::integral_constant<std::size_t, N>;
} // namespace alpaka

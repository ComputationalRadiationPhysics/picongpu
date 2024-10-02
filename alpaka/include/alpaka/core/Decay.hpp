/* Copyright 2023 Sergei Bastrakov, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"

#include <type_traits>

namespace alpaka
{
    //! Provides a decaying wrapper around std::is_same. Example: is_decayed_v<volatile float, float> returns true.
    template<typename T, typename U>
    inline constexpr auto is_decayed_v = std::is_same_v<std::decay_t<T>, std::decay_t<U>>;
} // namespace alpaka

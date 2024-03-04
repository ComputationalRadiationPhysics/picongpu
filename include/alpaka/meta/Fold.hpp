/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

namespace alpaka::meta
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TFnObj, typename T>
    ALPAKA_FN_HOST_ACC constexpr auto foldr(TFnObj const& /* f */, T const& t) -> T
    {
        return t;
    }

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TFnObj, typename T0, typename T1, typename... Ts>
    ALPAKA_FN_HOST_ACC constexpr auto foldr(TFnObj const& f, T0 const& t0, T1 const& t1, Ts const&... ts)
    {
        return f(t0, foldr(f, t1, ts...));
    }
} // namespace alpaka::meta

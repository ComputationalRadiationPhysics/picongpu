/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>

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

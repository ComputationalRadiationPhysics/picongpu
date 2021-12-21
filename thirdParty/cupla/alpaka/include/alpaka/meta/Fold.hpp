/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

namespace alpaka
{
    namespace meta
    {
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TFnObj, typename T>
        ALPAKA_FN_HOST_ACC auto foldr(TFnObj const& f, T const& t) -> T
        {
            alpaka::ignore_unused(f);

            return t;
        }
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TFnObj, typename T0, typename T1, typename... Ts>
        ALPAKA_FN_HOST_ACC auto foldr(TFnObj const& f, T0 const& t0, T1 const& t1, Ts const&... ts)
        {
            return f(t0, foldr(f, t1, ts...));
        }
    } // namespace meta
} // namespace alpaka

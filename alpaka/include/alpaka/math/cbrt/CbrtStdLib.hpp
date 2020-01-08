/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/cbrt/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library cbrt.
        class CbrtStdLib
        {
        public:
            using CbrtBase = CbrtStdLib;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library cbrt trait specialization.
            template<
                typename TArg>
            struct Cbrt<
                CbrtStdLib,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                ALPAKA_FN_HOST static auto cbrt(
                    CbrtStdLib const & cbrt_ctx,
                    TArg const & arg)
                -> decltype(std::cbrt(arg))
                {
                    alpaka::ignore_unused(cbrt_ctx);
                    return std::cbrt(arg);
                }
            };
        }
    }
}

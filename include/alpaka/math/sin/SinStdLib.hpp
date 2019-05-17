/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <alpaka/math/sin/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library sin.
        class SinStdLib
        {
        public:
            using SinBase = SinStdLib;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library sin trait specialization.
            template<
                typename TArg>
            struct Sin<
                SinStdLib,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                ALPAKA_FN_HOST static auto sin(
                    SinStdLib const & sin_ctx,
                    TArg const & arg)
                -> decltype(std::sin(arg))
                {
                    alpaka::ignore_unused(sin_ctx);
                    return std::sin(arg);
                }
            };
        }
    }
}

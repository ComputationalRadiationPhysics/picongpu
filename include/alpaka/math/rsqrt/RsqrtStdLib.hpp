/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <alpaka/math/rsqrt/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library rsqrt.
        class RsqrtStdLib
        {
        public:
            using RsqrtBase = RsqrtStdLib;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library rsqrt trait specialization.
            template<
                typename TArg>
            struct Rsqrt<
                RsqrtStdLib,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                ALPAKA_FN_HOST static auto rsqrt(
                    RsqrtStdLib const & rsqrt_ctx,
                    TArg const & arg)
                -> decltype(std::sqrt(arg))
                {
                    alpaka::ignore_unused(rsqrt_ctx);
                    return static_cast<TArg>(1)/std::sqrt(arg);
                }
            };
        }
    }
}

/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/asin/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library asin.
        class AsinStdLib
        {
        public:
            using AsinBase = AsinStdLib;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library asin trait specialization.
            template<
                typename TArg>
            struct Asin<
                AsinStdLib,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                ALPAKA_FN_HOST static auto asin(
                    AsinStdLib const & asin_ctx,
                    TArg const & arg)
                -> decltype(std::asin(arg))
                {
                    alpaka::ignore_unused(asin_ctx);
                    return std::asin(arg);
                }
            };
        }
    }
}

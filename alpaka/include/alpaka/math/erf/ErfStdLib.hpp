/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/erf/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library erf.
        class ErfStdLib
        {
        public:
            using ErfBase = ErfStdLib;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library erf trait specialization.
            template<
                typename TArg>
            struct Erf<
                ErfStdLib,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                ALPAKA_FN_HOST static auto erf(
                    ErfStdLib const & erf_ctx,
                    TArg const & arg)
                -> decltype(std::erf(arg))
                {
                    alpaka::ignore_unused(erf_ctx);
                    return std::erf(arg);
                }
            };
        }
    }
}

/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/acos/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library acos.
        class AcosStdLib
        {
        public:
            using AcosBase = AcosStdLib;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library acos trait specialization.
            template<
                typename TArg>
            struct Acos<
                AcosStdLib,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                ALPAKA_FN_HOST static auto acos(
                    AcosStdLib const & acos_ctx,
                    TArg const & arg)
                -> decltype(std::acos(arg))
                {
                    alpaka::ignore_unused(acos_ctx);
                    return std::acos(arg);
                }
            };
        }
    }
}

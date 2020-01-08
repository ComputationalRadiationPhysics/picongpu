/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/fmod/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library fmod.
        class FmodStdLib
        {
        public:
            using FmodBase = FmodStdLib;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library fmod trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Fmod<
                FmodStdLib,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_arithmetic<Tx>::value
                    && std::is_arithmetic<Ty>::value>::type>
            {
                ALPAKA_FN_HOST static auto fmod(
                    FmodStdLib const & fmod_ctx,
                    Tx const & x,
                    Ty const & y)
                -> decltype(std::fmod(x, y))
                {
                    alpaka::ignore_unused(fmod_ctx);
                    return std::fmod(x, y);
                }
            };
        }
    }
}

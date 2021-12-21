/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Unused.hpp>
#include <alpaka/math/round/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library round.
        class RoundStdLib : public concepts::Implements<ConceptMathRound, RoundStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library round trait specialization.
            template<typename TArg>
            struct Round<RoundStdLib, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST static auto round(RoundStdLib const& round_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(round_ctx);
                    return std::round(arg);
                }
            };
            //#############################################################################
            //! The standard library round trait specialization.
            template<typename TArg>
            struct Lround<RoundStdLib, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST static auto lround(RoundStdLib const& lround_ctx, TArg const& arg) -> long int
                {
                    alpaka::ignore_unused(lround_ctx);
                    return std::lround(arg);
                }
            };
            //#############################################################################
            //! The standard library round trait specialization.
            template<typename TArg>
            struct Llround<RoundStdLib, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST static auto llround(RoundStdLib const& llround_ctx, TArg const& arg) -> long int
                {
                    alpaka::ignore_unused(llround_ctx);
                    return std::llround(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

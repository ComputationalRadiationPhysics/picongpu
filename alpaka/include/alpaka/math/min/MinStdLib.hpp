/* Copyright 2019 Alexander Matthes, Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Unused.hpp>
#include <alpaka/math/min/Traits.hpp>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library min.
        class MinStdLib : public concepts::Implements<ConceptMathMin, MinStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library integral min trait specialization.
            template<typename Tx, typename Ty>
            struct Min<MinStdLib, Tx, Ty, std::enable_if_t<std::is_integral<Tx>::value && std::is_integral<Ty>::value>>
            {
                ALPAKA_FN_HOST static auto min(MinStdLib const& min_ctx, Tx const& x, Ty const& y)
                {
                    alpaka::ignore_unused(min_ctx);
                    return std::min(x, y);
                }
            };
            //#############################################################################
            //! The standard library mixed integral floating point min trait specialization.
            template<typename Tx, typename Ty>
            struct Min<
                MinStdLib,
                Tx,
                Ty,
                std::enable_if_t<
                    std::is_arithmetic<Tx>::value && std::is_arithmetic<Ty>::value
                    && !(std::is_integral<Tx>::value && std::is_integral<Ty>::value)>>
            {
                ALPAKA_FN_HOST static auto min(MinStdLib const& min_ctx, Tx const& x, Ty const& y)
                {
                    alpaka::ignore_unused(min_ctx);
                    return std::fmin(x, y);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

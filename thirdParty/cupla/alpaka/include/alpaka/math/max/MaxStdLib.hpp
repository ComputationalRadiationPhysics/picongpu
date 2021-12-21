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
#include <alpaka/math/max/Traits.hpp>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library max.
        class MaxStdLib : public concepts::Implements<ConceptMathMax, MaxStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library integral max trait specialization.
            template<typename Tx, typename Ty>
            struct Max<MaxStdLib, Tx, Ty, std::enable_if_t<std::is_integral<Tx>::value && std::is_integral<Ty>::value>>
            {
                ALPAKA_FN_HOST static auto max(MaxStdLib const& max_ctx, Tx const& x, Ty const& y)
                {
                    alpaka::ignore_unused(max_ctx);
                    return std::max(x, y);
                }
            };
            //#############################################################################
            //! The standard library mixed integral floating point max trait specialization.
            template<typename Tx, typename Ty>
            struct Max<
                MaxStdLib,
                Tx,
                Ty,
                std::enable_if_t<
                    std::is_arithmetic<Tx>::value && std::is_arithmetic<Ty>::value
                    && !(std::is_integral<Tx>::value && std::is_integral<Ty>::value)>>
            {
                ALPAKA_FN_HOST static auto max(MaxStdLib const& max_ctx, Tx const& x, Ty const& y)
                {
                    alpaka::ignore_unused(max_ctx);
                    return std::fmax(x, y);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

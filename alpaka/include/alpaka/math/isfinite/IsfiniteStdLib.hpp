/* Copyright 2021 Axel Huebl, Benjamin Worpitz, Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Unused.hpp>
#include <alpaka/math/isfinite/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The standard library isfinite.
        class IsfiniteStdLib : public concepts::Implements<ConceptMathIsfinite, IsfiniteStdLib>
        {
        };

        namespace traits
        {
            //! The standard library isfinite trait specialization.
            template<typename TArg>
            struct Isfinite<IsfiniteStdLib, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST auto operator()(IsfiniteStdLib const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    return std::isfinite(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

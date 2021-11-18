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
#include <alpaka/math/isnan/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The standard library isnan.
        class IsnanStdLib : public concepts::Implements<ConceptMathIsnan, IsnanStdLib>
        {
        };

        namespace traits
        {
            //! The standard library isnan trait specialization.
            template<typename TArg>
            struct Isnan<IsnanStdLib, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST auto operator()(IsnanStdLib const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    return std::isnan(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

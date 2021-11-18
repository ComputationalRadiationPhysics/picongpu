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
#include <alpaka/math/isinf/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The standard library isinf.
        class IsinfStdLib : public concepts::Implements<ConceptMathIsinf, IsinfStdLib>
        {
        };

        namespace traits
        {
            //! The standard library isinf trait specialization.
            template<typename TArg>
            struct Isinf<IsinfStdLib, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST auto operator()(IsinfStdLib const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    return std::isinf(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

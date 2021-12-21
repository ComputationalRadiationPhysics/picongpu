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
#include <alpaka/math/ceil/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library ceil.
        class CeilStdLib : public concepts::Implements<ConceptMathCeil, CeilStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library ceil trait specialization.
            template<typename TArg>
            struct Ceil<CeilStdLib, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST static auto ceil(CeilStdLib const& ceil_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ceil_ctx);
                    return std::ceil(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

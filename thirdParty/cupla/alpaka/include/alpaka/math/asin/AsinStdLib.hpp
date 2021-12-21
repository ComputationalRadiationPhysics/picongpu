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
#include <alpaka/math/asin/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library asin.
        class AsinStdLib : public concepts::Implements<ConceptMathAsin, AsinStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library asin trait specialization.
            template<typename TArg>
            struct Asin<AsinStdLib, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST static auto asin(AsinStdLib const& asin_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(asin_ctx);
                    return std::asin(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

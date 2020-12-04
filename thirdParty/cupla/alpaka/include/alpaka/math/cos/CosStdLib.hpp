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
#include <alpaka/math/cos/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library cos.
        class CosStdLib : public concepts::Implements<ConceptMathCos, CosStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library cos trait specialization.
            template<typename TArg>
            struct Cos<CosStdLib, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST static auto cos(CosStdLib const& cos_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(cos_ctx);
                    return std::cos(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

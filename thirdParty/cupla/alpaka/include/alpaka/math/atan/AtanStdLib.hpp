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
#include <alpaka/math/atan/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library atan.
        class AtanStdLib : public concepts::Implements<ConceptMathAtan, AtanStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library atan trait specialization.
            template<typename TArg>
            struct Atan<AtanStdLib, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST static auto atan(AtanStdLib const& atan_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(atan_ctx);
                    return std::atan(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

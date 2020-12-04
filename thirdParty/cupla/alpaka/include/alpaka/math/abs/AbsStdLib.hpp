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
#include <alpaka/math/abs/Traits.hpp>

#include <cmath>
#include <cstdlib>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library abs.
        class AbsStdLib : public concepts::Implements<ConceptMathAbs, AbsStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library abs trait specialization.
            template<typename TArg>
            struct Abs<
                AbsStdLib,
                TArg,
                std::enable_if_t<std::is_arithmetic<TArg>::value && std::is_signed<TArg>::value>>
            {
                ALPAKA_FN_HOST static auto abs(AbsStdLib const& abs_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(abs_ctx);
                    return std::abs(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

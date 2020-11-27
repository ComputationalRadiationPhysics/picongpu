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
#include <alpaka/math/pow/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library pow.
        class PowStdLib : public concepts::Implements<ConceptMathPow, PowStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library pow trait specialization.
            template<typename TBase, typename TExp>
            struct Pow<
                PowStdLib,
                TBase,
                TExp,
                std::enable_if_t<std::is_arithmetic<TBase>::value && std::is_arithmetic<TExp>::value>>
            {
                ALPAKA_FN_HOST static auto pow(PowStdLib const& pow_ctx, TBase const& base, TExp const& exp)
                {
                    alpaka::ignore_unused(pow_ctx);
                    return std::pow(base, exp);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

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
#include <alpaka/math/erf/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library erf.
        class ErfStdLib : public concepts::Implements<ConceptMathErf, ErfStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library erf trait specialization.
            template<typename TArg>
            struct Erf<ErfStdLib, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST static auto erf(ErfStdLib const& erf_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(erf_ctx);
                    return std::erf(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

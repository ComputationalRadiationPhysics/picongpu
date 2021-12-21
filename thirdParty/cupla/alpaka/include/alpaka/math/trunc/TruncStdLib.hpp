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
#include <alpaka/math/trunc/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library trunc.
        class TruncStdLib : public concepts::Implements<ConceptMathTrunc, TruncStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library trunc trait specialization.
            template<typename TArg>
            struct Trunc<TruncStdLib, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST static auto trunc(TruncStdLib const& trunc_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(trunc_ctx);
                    return std::trunc(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

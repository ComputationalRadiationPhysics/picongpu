/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Unused.hpp>
#include <alpaka/math/sincos/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library sincos.
        class SinCosStdLib : public concepts::Implements<ConceptMathSinCos, SinCosStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library sincos trait specialization.
            template<typename TArg>
            struct SinCos<SinCosStdLib, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                ALPAKA_FN_HOST static auto sincos(
                    SinCosStdLib const& sincos_ctx,
                    TArg const& arg,
                    TArg& result_sin,
                    TArg& result_cos) -> void
                {
                    alpaka::ignore_unused(sincos_ctx);
                    result_sin = std::sin(arg);
                    result_cos = std::cos(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

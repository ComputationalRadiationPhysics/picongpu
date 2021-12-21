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
#include <alpaka/math/fmod/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library fmod.
        class FmodStdLib : public concepts::Implements<ConceptMathFmod, FmodStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library fmod trait specialization.
            template<typename Tx, typename Ty>
            struct Fmod<
                FmodStdLib,
                Tx,
                Ty,
                std::enable_if_t<std::is_arithmetic<Tx>::value && std::is_arithmetic<Ty>::value>>
            {
                ALPAKA_FN_HOST static auto fmod(FmodStdLib const& fmod_ctx, Tx const& x, Ty const& y)
                {
                    alpaka::ignore_unused(fmod_ctx);
                    return std::fmod(x, y);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

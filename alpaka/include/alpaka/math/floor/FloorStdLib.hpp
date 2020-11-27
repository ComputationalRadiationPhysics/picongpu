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
#include <alpaka/math/floor/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library floor.
        class FloorStdLib : public concepts::Implements<ConceptMathFloor, FloorStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library floor trait specialization.
            template<typename TArg>
            struct Floor<FloorStdLib, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                ALPAKA_FN_HOST static auto floor(FloorStdLib const& floor_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(floor_ctx);
                    return std::floor(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

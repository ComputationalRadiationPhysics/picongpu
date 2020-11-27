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
#include <alpaka/math/atan2/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library atan2.
        class Atan2StdLib : public concepts::Implements<ConceptMathAtan2, Atan2StdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library atan2 trait specialization.
            template<typename Ty, typename Tx>
            struct Atan2<
                Atan2StdLib,
                Ty,
                Tx,
                std::enable_if_t<std::is_arithmetic<Ty>::value && std::is_arithmetic<Tx>::value>>
            {
                ALPAKA_FN_HOST static auto atan2(Atan2StdLib const& abs, Ty const& y, Tx const& x)
                {
                    alpaka::ignore_unused(abs);
                    return std::atan2(y, x);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

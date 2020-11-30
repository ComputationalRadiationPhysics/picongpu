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
#include <alpaka/math/remainder/Traits.hpp>

#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library remainder.
        class RemainderStdLib : public concepts::Implements<ConceptMathRemainder, RemainderStdLib>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library remainder trait specialization.
            template<typename Tx, typename Ty>
            struct Remainder<
                RemainderStdLib,
                Tx,
                Ty,
                std::enable_if_t<std::is_floating_point<Tx>::value && std::is_floating_point<Ty>::value>>
            {
                ALPAKA_FN_HOST static auto remainder(RemainderStdLib const& remainder_ctx, Tx const& x, Ty const& y)
                {
                    alpaka::ignore_unused(remainder_ctx);
                    return std::remainder(x, y);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        struct ConceptMathRemainder
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The remainder trait.
            template<typename T, typename Tx, typename Ty, typename TSfinae = void>
            struct Remainder;
        } // namespace traits

        //-----------------------------------------------------------------------------
        //! Computes the IEEE remainder of the floating point division operation x/y.
        //!
        //! \tparam T The type of the object specializing Remainder.
        //! \tparam Tx The type of the first argument.
        //! \tparam Ty The type of the second argument.
        //! \param remainder_ctx The object specializing Max.
        //! \param x The first argument.
        //! \param y The second argument.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename Tx, typename Ty>
        ALPAKA_FN_HOST_ACC auto remainder(T const& remainder_ctx, Tx const& x, Ty const& y)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathRemainder, T>;
            return traits::Remainder<ImplementationBase, Tx, Ty>::remainder(remainder_ctx, x, y);
        }
    } // namespace math
} // namespace alpaka

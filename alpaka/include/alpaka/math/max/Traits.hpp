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
        struct ConceptMathMax
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The max trait.
            template<typename T, typename Tx, typename Ty, typename TSfinae = void>
            struct Max;
        } // namespace traits

        //-----------------------------------------------------------------------------
        //! Returns the larger of two arguments.
        //! NaNs are treated as missing data (between a NaN and a numeric value, the numeric value is chosen).
        //!
        //! \tparam T The type of the object specializing Max.
        //! \tparam Tx The type of the first argument.
        //! \tparam Ty The type of the second argument.
        //! \param max_ctx The object specializing Max.
        //! \param x The first argument.
        //! \param y The second argument.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename Tx, typename Ty>
        ALPAKA_FN_HOST_ACC auto max(T const& max_ctx, Tx const& x, Ty const& y)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathMax, T>;
            return traits::Max<ImplementationBase, Tx, Ty>::max(max_ctx, x, y);
        }
    } // namespace math
} // namespace alpaka

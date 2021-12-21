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
        struct ConceptMathAtan2
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The atan2 trait.
            template<typename T, typename Ty, typename Tx, typename TSfinae = void>
            struct Atan2;
        } // namespace traits

        //-----------------------------------------------------------------------------
        //! Computes the arc tangent of y/x using the signs of arguments to determine the correct quadrant.
        //!
        //! \tparam T The type of the object specializing Atan2.
        //! \tparam Ty The y arg type.
        //! \tparam Tx The x arg type.
        //! \param atan2_ctx The object specializing Atan2.
        //! \param y The y arg.
        //! \param x The x arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename Ty, typename Tx>
        ALPAKA_FN_HOST_ACC auto atan2(T const& atan2_ctx, Ty const& y, Tx const& x)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathAtan2, T>;
            return traits::Atan2<ImplementationBase, Ty, Tx>::atan2(atan2_ctx, y, x);
        }
    } // namespace math
} // namespace alpaka

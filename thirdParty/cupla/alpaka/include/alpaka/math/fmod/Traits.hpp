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
        struct ConceptMathFmod
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The fmod trait.
            template<typename T, typename Tx, typename Ty, typename TSfinae = void>
            struct Fmod;
        } // namespace traits

        //-----------------------------------------------------------------------------
        //! Computes the floating-point remainder of the division operation x/y.
        //!
        //! \tparam T The type of the object specializing Fmod.
        //! \tparam Tx The type of the first argument.
        //! \tparam Ty The type of the second argument.
        //! \param fmod_ctx The object specializing Fmod.
        //! \param x The first argument.
        //! \param y The second argument.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename Tx, typename Ty>
        ALPAKA_FN_HOST_ACC auto fmod(T const& fmod_ctx, Tx const& x, Ty const& y)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathFmod, T>;
            return traits::Fmod<ImplementationBase, Tx, Ty>::fmod(fmod_ctx, x, y);
        }
    } // namespace math
} // namespace alpaka

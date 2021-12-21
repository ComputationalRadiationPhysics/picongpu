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
        struct ConceptMathRound
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The round trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Round;

            //#############################################################################
            //! The round trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Lround;

            //#############################################################################
            //! The round trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Llround;
        } // namespace traits

        //-----------------------------------------------------------------------------
        //! Computes the nearest integer value to arg (in floating-point format), rounding halfway cases away from
        //! zero, regardless of the current rounding mode.
        //!
        //! \tparam T The type of the object specializing Round.
        //! \tparam TArg The arg type.
        //! \param round_ctx The object specializing Round.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto round(T const& round_ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathRound, T>;
            return traits::Round<ImplementationBase, TArg>::round(round_ctx, arg);
        }
        //-----------------------------------------------------------------------------
        //! Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero,
        //! regardless of the current rounding mode.
        //!
        //! \tparam T The type of the object specializing Round.
        //! \tparam TArg The arg type.
        //! \param lround_ctx The object specializing Round.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto lround(T const& lround_ctx, TArg const& arg) -> long int
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathRound, T>;
            return traits::Lround<ImplementationBase, TArg>::lround(lround_ctx, arg);
        }
        //-----------------------------------------------------------------------------
        //! Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero,
        //! regardless of the current rounding mode.
        //!
        //! \tparam T The type of the object specializing Round.
        //! \tparam TArg The arg type.
        //! \param llround_ctx The object specializing Round.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto llround(T const& llround_ctx, TArg const& arg) -> long long int
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathRound, T>;
            return traits::Llround<ImplementationBase, TArg>::llround(llround_ctx, arg);
        }
    } // namespace math
} // namespace alpaka

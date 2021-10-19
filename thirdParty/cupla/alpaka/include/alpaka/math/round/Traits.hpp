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
#include <alpaka/core/Unused.hpp>

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
            //! The round trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Round
            {
                ALPAKA_FN_HOST_ACC auto operator()(T const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    // This is an ADL call. If you get a compile error here then your type is not supported by the
                    // backend and we could not find round(TArg) in the namespace of your type.
                    return round(arg);
                }
            };

            //! The round trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Lround
            {
                ALPAKA_FN_HOST_ACC auto operator()(T const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    // This is an ADL call. If you get a compile error here then your type is not supported by the
                    // backend and we could not find lround(TArg) in the namespace of your type.
                    return lround(arg);
                }
            };

            //! The round trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Llround
            {
                ALPAKA_FN_HOST_ACC auto operator()(T const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    // This is an ADL call. If you get a compile error here then your type is not supported by the
                    // backend and we could not find llround(TArg) in the namespace of your type.
                    return llround(arg);
                }
            };
        } // namespace traits

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
            return traits::Round<ImplementationBase, TArg>{}(round_ctx, arg);
        }
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
            return traits::Lround<ImplementationBase, TArg>{}(lround_ctx, arg);
        }
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
            return traits::Llround<ImplementationBase, TArg>{}(llround_ctx, arg);
        }
    } // namespace math
} // namespace alpaka

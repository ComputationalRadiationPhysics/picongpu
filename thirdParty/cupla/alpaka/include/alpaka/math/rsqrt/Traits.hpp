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
        struct ConceptMathRsqrt
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The rsqrt trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Rsqrt;
        } // namespace traits

        //-----------------------------------------------------------------------------
        //! Computes the rsqrt.
        //!
        //! Valid real arguments are positive. For other values the result
        //! may depend on the backend and compilation options, will likely
        //! be NaN.
        //!
        //! \tparam T The type of the object specializing Rsqrt.
        //! \tparam TArg The arg type.
        //! \param rsqrt_ctx The object specializing Rsqrt.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto rsqrt(T const& rsqrt_ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathRsqrt, T>;
            return traits::Rsqrt<ImplementationBase, TArg>::rsqrt(rsqrt_ctx, arg);
        }
    } // namespace math
} // namespace alpaka

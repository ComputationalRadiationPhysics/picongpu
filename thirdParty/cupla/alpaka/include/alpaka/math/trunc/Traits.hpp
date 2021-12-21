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
        struct ConceptMathTrunc
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The trunc trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Trunc;
        } // namespace traits

        //-----------------------------------------------------------------------------
        //! Computes the nearest integer not greater in magnitude than arg.
        //!
        //! \tparam T The type of the object specializing Trunc.
        //! \tparam TArg The arg type.
        //! \param trunc_ctx The object specializing Trunc.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto trunc(T const& trunc_ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathTrunc, T>;
            return traits::Trunc<ImplementationBase, TArg>::trunc(trunc_ctx, arg);
        }
    } // namespace math
} // namespace alpaka

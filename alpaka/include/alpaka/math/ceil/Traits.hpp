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
        struct ConceptMathCeil
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The ceil trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Ceil;
        } // namespace traits

        //-----------------------------------------------------------------------------
        //! Computes the smallest integer value not less than arg.
        //!
        //! \tparam T The type of the object specializing Ceil.
        //! \tparam TArg The arg type.
        //! \param ceil_ctx The object specializing Ceil.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto ceil(T const& ceil_ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathCeil, T>;
            return traits::Ceil<ImplementationBase, TArg>::ceil(ceil_ctx, arg);
        }
    } // namespace math
} // namespace alpaka

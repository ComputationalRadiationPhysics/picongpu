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
        struct ConceptMathErf
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The erf trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Erf;
        } // namespace traits

        //-----------------------------------------------------------------------------
        //! Computes the error function of arg.
        //!
        //! \tparam T The type of the object specializing Erf.
        //! \tparam TArg The arg type.
        //! \param erf_ctx The object specializing Erf.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto erf(T const& erf_ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathErf, T>;
            return traits::Erf<ImplementationBase, TArg>::erf(erf_ctx, arg);
        }
    } // namespace math
} // namespace alpaka

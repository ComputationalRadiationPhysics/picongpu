/* Copyright 2019 Benjamin Worpitz, Matthias Werner
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
        struct ConceptMathSinCos
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The sincos trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct SinCos;
        } // namespace traits

        //-----------------------------------------------------------------------------
        //! Computes the sine and cosine (measured in radians).
        //!
        //! \tparam T The type of the object specializing SinCos.
        //! \tparam TArg The arg type.
        //! \param sincos_ctx The object specializing SinCos.
        //! \param arg The arg.
        //! \param result_sin result of sine
        //! \param result_cos result of cosine
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto sincos(T const& sincos_ctx, TArg const& arg, TArg& result_sin, TArg& result_cos)
            -> void
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathSinCos, T>;
            traits::SinCos<ImplementationBase, TArg>::sincos(sincos_ctx, arg, result_sin, result_cos);
        }
    } // namespace math
} // namespace alpaka

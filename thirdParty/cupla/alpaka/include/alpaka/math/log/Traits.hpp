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
        struct ConceptMathLog
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The log trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Log;
        } // namespace traits

        //-----------------------------------------------------------------------------
        //! Computes the the natural (base e) logarithm of arg.
        //!
        //! Valid real arguments are non-negative. For other values the result
        //! may depend on the backend and compilation options, will likely
        //! be NaN.
        //!
        //! \tparam T The type of the object specializing Log.
        //! \tparam TArg The arg type.
        //! \param log_ctx The object specializing Log.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto log(T const& log_ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathLog, T>;
            return traits::Log<ImplementationBase, TArg>::log(log_ctx, arg);
        }
    } // namespace math
} // namespace alpaka

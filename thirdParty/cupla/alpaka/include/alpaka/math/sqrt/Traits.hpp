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
        struct ConceptMathSqrt
        {
        };

        namespace traits
        {
            //! The sqrt trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Sqrt
            {
                ALPAKA_FN_HOST_ACC auto operator()(T const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    // This is an ADL call. If you get a compile error here then your type is not supported by the
                    // backend and we could not find sqrt(TArg) in the namespace of your type.
                    return sqrt(arg);
                }
            };
        } // namespace traits

        //! Computes the square root of arg.
        //!
        //! Valid real arguments are non-negative. For other values the result
        //! may depend on the backend and compilation options, will likely
        //! be NaN.
        //!
        //! \tparam T The type of the object specializing Sqrt.
        //! \tparam TArg The arg type.
        //! \param sqrt_ctx The object specializing Sqrt.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto sqrt(T const& sqrt_ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathSqrt, T>;
            return traits::Sqrt<ImplementationBase, TArg>{}(sqrt_ctx, arg);
        }
    } // namespace math
} // namespace alpaka

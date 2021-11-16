/* Copyright 2021 Benjamin Worpitz, Jeffrey Kelling
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
        struct ConceptMathIsnan
        {
        };

        namespace traits
        {
            //! The exp trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Isnan
            {
                ALPAKA_FN_HOST_ACC auto operator()(T const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    // This is an ADL call. If you get a compile error here then your type is not supported by the
                    // backend and we could not find isnan(TArg) in the namespace of your type.
                    return isnan(arg);
                }
            };
        } // namespace traits

        //! Checks if given value is NaN.
        //!
        //! \tparam T The type of the object specializing Isnan.
        //! \tparam TArg The arg type.
        //! \param ctx The object specializing Isnan.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto isnan(T const& ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathIsnan, T>;
            return traits::Isnan<ImplementationBase, TArg>{}(ctx, arg);
        }
    } // namespace math
} // namespace alpaka

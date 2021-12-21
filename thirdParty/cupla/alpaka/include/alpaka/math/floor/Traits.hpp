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
        struct ConceptMathFloor
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The floor trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Floor;
        } // namespace traits

        //-----------------------------------------------------------------------------
        //! Computes the largest integer value not greater than arg.
        //!
        //! \tparam T The type of the object specializing Floor.
        //! \tparam TArg The arg type.
        //! \param floor_ctx The object specializing Floor.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto floor(T const& floor_ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathFloor, T>;
            return traits::Floor<ImplementationBase, TArg>::floor(floor_ctx, arg);
        }
    } // namespace math
} // namespace alpaka

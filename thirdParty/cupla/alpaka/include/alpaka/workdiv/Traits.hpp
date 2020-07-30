/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/idx/Traits.hpp>
#include <alpaka/core/Concepts.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Common.hpp>

#include <type_traits>
#include <utility>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The work division traits specifics.
    namespace workdiv
    {
        struct ConceptWorkDiv{};

        //-----------------------------------------------------------------------------
        //! The work division traits.
        namespace traits
        {
            //#############################################################################
            //! The work div trait.
            template<
                typename TWorkDiv,
                typename TOrigin,
                typename TUnit,
                typename TSfinae = void>
            struct GetWorkDiv;
        }

        //-----------------------------------------------------------------------------
        //! Get the extent requested.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOrigin,
            typename TUnit,
            typename TWorkDiv>
        ALPAKA_FN_HOST_ACC auto getWorkDiv(
            TWorkDiv const & workDiv)
        -> vec::Vec<dim::Dim<TWorkDiv>, idx::Idx<TWorkDiv>>
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptWorkDiv, TWorkDiv>;
            return
                traits::GetWorkDiv<
                    ImplementationBase,
                    TOrigin,
                    TUnit>
                ::getWorkDiv(
                    workDiv);
        }

        namespace traits
        {
            //#############################################################################
            //! The work div grid thread extent trait specialization.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
                {
                    return
                        workdiv::getWorkDiv<origin::Grid, unit::Blocks>(workDiv)
                        * workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv);
                }
            };
            //#############################################################################
            //! The work div grid element extent trait specialization.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Grid,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
                {
                    return
                        workdiv::getWorkDiv<origin::Grid, unit::Threads>(workDiv)
                        * workdiv::getWorkDiv<origin::Thread, unit::Elems>(workDiv);
                }
            };
            //#############################################################################
            //! The work div block element extent trait specialization.
            template<
                typename TWorkDiv>
            struct GetWorkDiv<
                TWorkDiv,
                origin::Block,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    TWorkDiv const & workDiv)
                {
                    return
                        workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv)
                        * workdiv::getWorkDiv<origin::Thread, unit::Elems>(workDiv);
                }
            };
        }
    }
}

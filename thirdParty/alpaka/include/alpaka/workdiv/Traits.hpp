/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Positioning.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <type_traits>
#include <utility>

namespace alpaka
{
    struct ConceptWorkDiv
    {
    };

    //! The work division trait.
    namespace trait
    {
        //! The work div trait.
        template<typename TWorkDiv, typename TOrigin, typename TUnit, typename TSfinae = void>
        struct GetWorkDiv;
    } // namespace trait

    //! Get the extent requested.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOrigin, typename TUnit, typename TWorkDiv>
    ALPAKA_FN_HOST_ACC auto getWorkDiv(TWorkDiv const& workDiv) -> Vec<Dim<TWorkDiv>, Idx<TWorkDiv>>
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptWorkDiv, TWorkDiv>;
        return trait::GetWorkDiv<ImplementationBase, TOrigin, TUnit>::getWorkDiv(workDiv);
    }

    namespace trait
    {
        //! The work div grid thread extent trait specialization.
        template<typename TWorkDiv>
        struct GetWorkDiv<TWorkDiv, origin::Grid, unit::Threads>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getWorkDiv(TWorkDiv const& workDiv)
            {
                return alpaka::getWorkDiv<origin::Grid, unit::Blocks>(workDiv)
                       * alpaka::getWorkDiv<origin::Block, unit::Threads>(workDiv);
            }
        };

        //! The work div grid element extent trait specialization.
        template<typename TWorkDiv>
        struct GetWorkDiv<TWorkDiv, origin::Grid, unit::Elems>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getWorkDiv(TWorkDiv const& workDiv)
            {
                return alpaka::getWorkDiv<origin::Grid, unit::Threads>(workDiv)
                       * alpaka::getWorkDiv<origin::Thread, unit::Elems>(workDiv);
            }
        };

        //! The work div block element extent trait specialization.
        template<typename TWorkDiv>
        struct GetWorkDiv<TWorkDiv, origin::Block, unit::Elems>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getWorkDiv(TWorkDiv const& workDiv)
            {
                return alpaka::getWorkDiv<origin::Block, unit::Threads>(workDiv)
                       * alpaka::getWorkDiv<origin::Thread, unit::Elems>(workDiv);
            }
        };
    } // namespace trait
} // namespace alpaka

/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan
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
    struct ConceptTime
    {
    };

    //! The time trait.
    namespace trait
    {
        //! The clock trait.
        template<typename TTime, typename TSfinae = void>
        struct Clock;
    } // namespace trait

    //! \return A counter that is increasing every clock cycle.
    //!
    //! \tparam TTime The time implementation type.
    //! \param time The time implementation.
    ALPAKA_NO_HOST_ACC_WARNING template<typename TTime>
    [[deprecated("clock() is deprecated and will be removed in the next release")]] ALPAKA_FN_HOST_ACC auto clock(
        TTime const& time) -> std::uint64_t
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptTime, TTime>;
        return trait::Clock<ImplementationBase>::clock(time);
    }
} // namespace alpaka

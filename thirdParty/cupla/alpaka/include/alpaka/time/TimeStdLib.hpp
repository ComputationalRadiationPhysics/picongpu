/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/time/Traits.hpp>

#include <chrono>

namespace alpaka
{
    //! The CPU fibers accelerator time implementation.
    class TimeStdLib : public concepts::Implements<ConceptTime, TimeStdLib>
    {
    };

    namespace trait
    {
        //! The CPU fibers accelerator clock operation.
        template<>
        struct Clock<TimeStdLib>
        {
            ALPAKA_FN_HOST static auto clock(TimeStdLib const& /* time */) -> std::uint64_t
            {
                // NOTE: high_resolution_clock returns a non-steady wall-clock time!
                // This means that it is not ensured that the values will always increase monotonically.
                return static_cast<std::uint64_t>(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count());
            }
        };
    } // namespace trait
} // namespace alpaka

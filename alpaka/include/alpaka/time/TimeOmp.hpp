/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef _OPENMP

#    include <alpaka/core/Common.hpp>
#    include <alpaka/time/Traits.hpp>

#    include <omp.h>

namespace alpaka
{
    //! The OpenMP accelerator time implementation.
    class TimeOmp : public concepts::Implements<ConceptTime, TimeOmp>
    {
    };

    namespace trait
    {
        //! The OpenMP accelerator clock operation.
        template<>
        struct Clock<TimeOmp>
        {
            ALPAKA_FN_HOST static auto clock(TimeOmp const& /* time */) -> std::uint64_t
            {
                // NOTE: We compute the number of clock ticks by dividing the following durations:
                // - omp_get_wtime returns the elapsed wall clock time in seconds.
                // - omp_get_wtick gets the timer precision, i.e., the number of seconds between two successive clock
                // ticks.
                return static_cast<std::uint64_t>(omp_get_wtime() / omp_get_wtick());
            }
        };
    } // namespace trait
} // namespace alpaka

#endif

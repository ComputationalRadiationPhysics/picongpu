/**
* \file
* Copyright 2016 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef _OPENMP

#include <alpaka/time/Traits.hpp>

#include <alpaka/core/Common.hpp>

#include <boost/core/ignore_unused.hpp>

#include <omp.h>

namespace alpaka
{
    namespace time
    {
        //#############################################################################
        //! The OpenMP accelerator time implementation.
        class TimeOmp
        {
        public:
            using TimeBase = TimeOmp;

            //-----------------------------------------------------------------------------
            TimeOmp() = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA TimeOmp(TimeOmp const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA TimeOmp(TimeOmp &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(TimeOmp const &) -> TimeOmp & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(TimeOmp &&) -> TimeOmp & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~TimeOmp() = default;
        };

        namespace traits
        {
            //#############################################################################
            //! The OpenMP accelerator clock operation.
            template<>
            struct Clock<
                time::TimeOmp>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA static auto clock(
                    time::TimeOmp const & time)
                -> std::uint64_t
                {
                    boost::ignore_unused(time);
                    // NOTE: We compute the number of clock ticks by dividing the following durations:
                    // - omp_get_wtime returns the elapsed wall clock time in seconds.
                    // - omp_get_wtick gets the timer precision, i.e., the number of seconds between two successive clock ticks. 
                    return
                        static_cast<std::uint64_t>(
                            omp_get_wtime() / omp_get_wtick());
                }
            };
        }
    }
}

#endif

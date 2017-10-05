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

#include <alpaka/time/Traits.hpp>       // time::Clock

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_*

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

//#include <ctime>                        // std::clock
#include <chrono>                       // std::chrono::high_resolution_clock

namespace alpaka
{
    namespace time
    {
        //#############################################################################
        //! The CPU fibers accelerator time implementation.
        //#############################################################################
        class TimeStl
        {
        public:
            using TimeBase = TimeStl;

            //-----------------------------------------------------------------------------
            //! Default constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA TimeStl() = default;
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA TimeStl(TimeStl const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA TimeStl(TimeStl &&) = delete;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(TimeStl const &) -> TimeStl & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(TimeStl &&) -> TimeStl & = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~TimeStl() = default;
        };

        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator clock operation.
            //#############################################################################
            template<>
            struct Clock<
                TimeStl>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA static auto clock(
                    time::TimeStl const & time)
                -> std::uint64_t
                {
                    boost::ignore_unused(time);

                    // NOTE: high_resolution_clock returns a non-steady wall-clock time!
                    // This means that it is not ensured that the values will always increase monotonically.
                    return
                        static_cast<std::uint64_t>(
                            std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                    .count());
                }
            };
        }
    }
}

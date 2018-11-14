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

#include <alpaka/time/Traits.hpp>

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

//#include <ctime>
#include <chrono>

namespace alpaka
{
    namespace time
    {
        //#############################################################################
        //! The CPU fibers accelerator time implementation.
        class TimeStdLib
        {
        public:
            using TimeBase = TimeStdLib;

            //-----------------------------------------------------------------------------
            TimeStdLib() = default;
            //-----------------------------------------------------------------------------
            TimeStdLib(TimeStdLib const &) = delete;
            //-----------------------------------------------------------------------------
            TimeStdLib(TimeStdLib &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(TimeStdLib const &) -> TimeStdLib & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(TimeStdLib &&) -> TimeStdLib & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~TimeStdLib() = default;
        };

        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator clock operation.
            template<>
            struct Clock<
                TimeStdLib>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto clock(
                    time::TimeStdLib const & time)
                -> std::uint64_t
                {
                    alpaka::ignore_unused(time);

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

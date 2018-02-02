/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#include <alpaka/alpaka.hpp>

#include <type_traits>
#include <utility>

namespace alpaka
{
    namespace test
    {
        namespace integ
        {
            //-----------------------------------------------------------------------------
            //! \return The run time of the given kernel.
            template<
                typename TStream,
                typename TExec>
            auto measureKernelRunTimeMs(
                TStream & stream,
                TExec && exec)
            -> std::chrono::milliseconds::rep
            {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                std::cout
                    << "measureKernelRunTime("
                    << " exec: " << typeid(typename std::decay<TExec>::type).name()
                    << " stream: " << typeid(TStream).name()
                    << ")" << std::endl;
#endif
                // Wait for the stream to finish all tasks enqueued prior to the kernel.
                alpaka::wait::wait(stream);

                // Take the time prior to the execution.
                auto const tpStart(std::chrono::high_resolution_clock::now());

                // Execute the kernel functor.
                alpaka::stream::enqueue(stream, std::forward<TExec>(exec));

                // Wait for the stream to finish the kernel execution to measure its run time.
                alpaka::wait::wait(stream);

                // Take the time after the execution.
                auto const tpEnd(std::chrono::high_resolution_clock::now());

                auto const durElapsed(tpEnd - tpStart);

                // Return the duration.
                return std::chrono::duration_cast<std::chrono::milliseconds>(durElapsed).count();
            }
        }
    }
}

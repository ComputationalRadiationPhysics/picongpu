/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
            template<typename TQueue, typename TTask>
            auto measureTaskRunTimeMs(TQueue& queue, TTask&& task) -> std::chrono::milliseconds::rep
            {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                std::cout << "measureKernelRunTime("
                          << " queue: " << typeid(TQueue).name() << " task: " << typeid(std::decay_t<TTask>).name()
                          << ")" << std::endl;
#endif
                // Wait for the queue to finish all tasks enqueued prior to the giventask.
                alpaka::wait(queue);

                // Take the time prior to the execution.
                auto const tpStart(std::chrono::high_resolution_clock::now());

                // Enqueue the task.
                alpaka::enqueue(queue, std::forward<TTask>(task));

                // Wait for the queue to finish the task execution to measure its run time.
                alpaka::wait(queue);

                // Take the time after the execution.
                auto const tpEnd(std::chrono::high_resolution_clock::now());

                auto const durElapsed(tpEnd - tpStart);

                // Return the duration.
                return std::chrono::duration_cast<std::chrono::milliseconds>(durElapsed).count();
            }
        } // namespace integ
    } // namespace test
} // namespace alpaka

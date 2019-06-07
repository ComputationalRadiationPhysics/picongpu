/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/event/EventCpu.hpp>

#include <alpaka/wait/Traits.hpp>

namespace alpaka
{
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device thread wait specialization.
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
            template<>
            struct CurrentThreadWaitFor<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    dev::DevCpu const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    // Get all the queues on the device at the time of invocation.
                    // All queues added afterwards are ignored.
                    auto vspQueues(
                        dev.m_spDevCpuImpl->GetAllAsyncQueueImpls());

                    // Enqueue an event in every asynchronous queue on the device.
                    // \FIXME: This should be done atomically for all queues.
                    // Furthermore there should not even be a chance to enqueue something between getting the queues and adding our wait events!
                    std::vector<event::EventCpu> vEvents;
                    for(auto && spQueue : vspQueues)
                    {
                        vEvents.emplace_back(dev);
                        queue::enqueue(spQueue, vEvents.back());
                    }

                    // Now wait for all the events.
                    for(auto && event : vEvents)
                    {
                        wait::wait(event);
                    }
                }
            };
        }
    }
}

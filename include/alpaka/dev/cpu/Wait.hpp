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
            //! Tasks that are enqueued or streams that are created after this call is made are not waited for.
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

                    // Get all the streams on the device at the time of invocation.
                    // All streams added afterwards are ignored.
                    auto vspStreams(
                        dev.m_spDevCpuImpl->GetAllAsyncStreamImpls());

                    // Enqueue an event in every asynchronous stream on the device.
                    // \FIXME: This should be done atomically for all streams.
                    // Furthermore there should not even be a chance to enqueue something between getting the streams and adding our wait events!
                    std::vector<event::EventCpu> vEvents;
                    for(auto && spStream : vspStreams)
                    {
                        vEvents.emplace_back(dev);
                        stream::enqueue(spStream, vEvents.back());
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

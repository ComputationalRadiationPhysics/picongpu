/* Copyright 2022 Benjamin Worpitz, Rene Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/event/EventCpu.hpp"
#include "alpaka/wait/Traits.hpp"

namespace alpaka::trait
{
    //! The CPU device thread wait specialization.
    //!
    //! Blocks until the device has completed all preceding requested tasks.
    //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
    template<>
    struct CurrentThreadWaitFor<DevCpu>
    {
        ALPAKA_FN_HOST static auto currentThreadWaitFor(DevCpu const& dev) -> void
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            generic::currentThreadWaitForDevice(dev);
        }
    };
} // namespace alpaka::trait

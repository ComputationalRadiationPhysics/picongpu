/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pmacc/eventSystem/queues/Queue.hpp"

#include "pmacc/alpakaHelper/Device.hpp"
#include "pmacc/alpakaHelper/acc.hpp"

#include <alpaka/alpaka.hpp>

namespace pmacc
{
    Queue::Queue() : queue(ComputeDeviceQueue(manager::Device<ComputeDevice>::get().current()))
    {
    }

    Queue::~Queue()
    {
        alpaka::wait(queue);
    }

    ComputeDeviceQueue Queue::getAlpakaQueue() const
    {
        return queue;
    }

    void Queue::waitOn(ComputeEventHandle const& ev)
    {
        if(queue != ev.getStream())
        {
            auto alpakaEvent = *ev;
            auto queue = this->getAlpakaQueue();
            alpaka::wait(queue, alpakaEvent);
        }
    }
} // namespace pmacc

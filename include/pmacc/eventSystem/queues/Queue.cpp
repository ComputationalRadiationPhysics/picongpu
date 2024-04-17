/* Copyright 2013-2023 Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "pmacc/eventSystem/queues/Queue.hpp"

#include "pmacc/alpakaHelper/Device.hpp"
#include "pmacc/alpakaHelper/acc.hpp"

#include <alpaka/alpaka.hpp>

namespace pmacc
{
    Queue::Queue() : queue(AccStream(manager::Device<ComputeDevice>::get().current()))
    {
    }

    Queue::~Queue()
    {
        alpaka::wait(queue);
    }

    AccStream Queue::getAlpakaQueue() const
    {
        return queue;
    }

    void Queue::waitOn(const CudaEventHandle& ev)
    {
        if(queue != ev.getStream())
        {
            auto alpakaEvent = *ev;
            auto queue = this->getAlpakaQueue();
            alpaka::wait(queue, alpakaEvent);
        }
    }
} // namespace pmacc

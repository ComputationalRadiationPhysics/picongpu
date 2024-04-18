/* Copyright 2016-2023 Rene Widera
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


#include "pmacc/eventSystem/events/ComputeEvent.hpp"

#include "pmacc/Environment.hpp"
#include "pmacc/alpakaHelper/Device.hpp"
#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/eventSystem/events/ComputeEventHandle.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    ComputeEvent::ComputeEvent() : event(ComputeDeviceEvent(manager::Device<ComputeDevice>::get().current()))
    {
        log(ggLog::CUDA_RT() + ggLog::EVENT(), "create event");
    }


    ComputeEvent::~ComputeEvent()
    {
        PMACC_ASSERT(refCounter == 0u);
        log(ggLog::CUDA_RT() + ggLog::EVENT(), "sync and delete event");
        alpaka::wait(event);
    }

    void ComputeEvent::registerHandle()
    {
        ++refCounter;
    }

    void ComputeEvent::releaseHandle()
    {
        assert(refCounter != 0u);
        // get old value and decrement
        uint32_t oldCounter = refCounter--;
        if(oldCounter == 1u)
        {
            // reset event meta data
            stream.reset();
            finished = true;

            Environment<>::get().EventPool().push(this);
        }
    }


    bool ComputeEvent::isFinished()
    {
        // avoid alpaka calls if event is already finished
        if(!finished)
        {
            assert(stream.has_value());
            finished = alpaka::isComplete(event);
        }
        return finished;
    }


    void ComputeEvent::recordEvent(ComputeDeviceQueue const& stream)
    {
        /* disallow double recording */
        assert(!this->stream.has_value());
        finished = false;
        this->stream = stream;
        alpaka::enqueue(*this->stream, event);
    }

} // namespace pmacc

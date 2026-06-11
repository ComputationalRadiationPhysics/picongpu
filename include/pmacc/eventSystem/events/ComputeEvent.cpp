/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

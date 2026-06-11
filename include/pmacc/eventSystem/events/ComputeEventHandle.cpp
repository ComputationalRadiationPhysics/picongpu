/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pmacc/eventSystem/events/ComputeEventHandle.hpp"

#include "pmacc/alpakaHelper/acc.hpp"

namespace pmacc
{
    ComputeEventHandle::ComputeEventHandle(ComputeEvent* const evPointer) : event(evPointer)
    {
        event->registerHandle();
    }

    ComputeEventHandle::ComputeEventHandle(ComputeEventHandle const& other)
    {
        /* register and release handle is done by the assign operator */
        *this = other;
    }

    ComputeEventHandle& ComputeEventHandle::operator=(ComputeEventHandle const& other)
    {
        /* check if an old event is overwritten */
        if(event)
            event->releaseHandle();
        event = other.event;
        /* check that new event pointer is not nullptr */
        if(event)
            event->registerHandle();
        return *this;
    }

    ComputeEventHandle::~ComputeEventHandle()
    {
        if(event)
            event->releaseHandle();
        event = nullptr;
    }

    ComputeDeviceEvent ComputeEventHandle::operator*() const
    {
        assert(event);
        return **event;
    }

    bool ComputeEventHandle::isFinished()
    {
        PMACC_ASSERT(event);
        return event->isFinished();
    }

    ComputeDeviceQueue ComputeEventHandle::getStream() const
    {
        PMACC_ASSERT(event);
        return event->getStream();
    }

    void ComputeEventHandle::recordEvent(ComputeDeviceQueue const& stream)
    {
        PMACC_ASSERT(event);
        event->recordEvent(stream);
    }

} // namespace pmacc

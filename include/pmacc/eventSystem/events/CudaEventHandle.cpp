/* Copyright 2014-2023 Rene Widera
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

#include "pmacc/eventSystem/events/CudaEventHandle.hpp"

#include "pmacc/alpakaHelper/acc.hpp"

namespace pmacc
{
    CudaEventHandle::CudaEventHandle(CudaEvent* const evPointer) : event(evPointer)
    {
        event->registerHandle();
    }

    CudaEventHandle::CudaEventHandle(const CudaEventHandle& other)
    {
        /* register and release handle is done by the assign operator */
        *this = other;
    }

    CudaEventHandle& CudaEventHandle::operator=(const CudaEventHandle& other)
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

    CudaEventHandle::~CudaEventHandle()
    {
        if(event)
            event->releaseHandle();
        event = nullptr;
    }

    AlpakaEventType CudaEventHandle::operator*() const
    {
        assert(event);
        return **event;
    }

    bool CudaEventHandle::isFinished()
    {
        PMACC_ASSERT(event);
        return event->isFinished();
    }

    AccStream CudaEventHandle::getStream() const
    {
        PMACC_ASSERT(event);
        return event->getStream();
    }

    void CudaEventHandle::recordEvent(AccStream const& stream)
    {
        PMACC_ASSERT(event);
        event->recordEvent(stream);
    }

} // namespace pmacc

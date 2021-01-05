/* Copyright 2014-2021 Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "pmacc/types.hpp"
#include "pmacc/assert.hpp"

#include "pmacc/eventSystem/events/CudaEvent.def"

namespace pmacc
{
    /** handle to CudaEvent */
    class CudaEventHandle
    {
    private:
        /** pointer to the CudaEvent */
        CudaEvent* event;

    public:
        /** create invalid handle  */
        CudaEventHandle() : event(nullptr)
        {
        }

        /** create a handle to a valid CudaEvent
         *
         * @param evPointer pointer to a CudaEvent
         */
        CudaEventHandle(CudaEvent* const evPointer) : event(evPointer)
        {
            event->registerHandle();
        }

        CudaEventHandle(const CudaEventHandle& other) : event(nullptr)
        {
            /* register and release handle is done by the assign operator */
            *this = other;
        }

        /** assign an event handle
         *
         * undefined behavior if the other event handle is equal to this instance
         *
         * @param other event handle
         * @return this handle
         */
        CudaEventHandle& operator=(const CudaEventHandle& other)
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

        /** Destructor */
        ~CudaEventHandle()
        {
            if(event)
                event->releaseHandle();
            event = nullptr;
        }

        /**
         * get native cupla event
         *
         * @return native cupla event
         */
        cuplaEvent_t operator*() const
        {
            assert(event);
            return **event;
        }

        /** check whether the event is finished
         *
         * @return true if event is finished else false
         */
        bool isFinished()
        {
            PMACC_ASSERT(event);
            return event->isFinished();
        }


        /** get stream in which this event is recorded
         *
         * @return native cupla stream
         */
        cuplaStream_t getStream() const
        {
            PMACC_ASSERT(event);
            return event->getStream();
        }

        /** record event in a device stream
         *
         * @param stream native cupla stream
         */
        void recordEvent(cuplaStream_t stream)
        {
            PMACC_ASSERT(event);
            event->recordEvent(stream);
        }
    };
} // namespace pmacc

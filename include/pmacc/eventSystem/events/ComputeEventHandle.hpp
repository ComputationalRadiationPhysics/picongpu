/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/assert.hpp"
#include "pmacc/eventSystem/events/ComputeEvent.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /** handle to ComputeEvent */
    class ComputeEventHandle
    {
    private:
        /** pointer to the ComputeEvent */
        ComputeEvent* event = nullptr;

    public:
        /** create invalid handle  */
        ComputeEventHandle() = default;

        /** create a handle to a valid ComputeEvent
         *
         * @param evPointer pointer to a ComputeEvent
         */
        ComputeEventHandle(ComputeEvent* const evPointer);

        ComputeEventHandle(ComputeEventHandle const& other);

        /** assign an event handle
         *
         * undefined behavior if the other event handle is equal to this instance
         *
         * @param other event handle
         * @return this handle
         */
        ComputeEventHandle& operator=(ComputeEventHandle const& other);

        /** Destructor */
        ~ComputeEventHandle();

        /**
         * get native alpaka event
         *
         * @return native alpaka event
         */
        ComputeDeviceEvent operator*() const;

        /** check whether the event is finished
         *
         * @return true if event is finished else false
         */
        bool isFinished();


        /** get stream in which this event is recorded
         *
         * @return native alpaka queue
         */
        ComputeDeviceQueue getStream() const;

        /** record event in a device queue
         *
         * @param stream native alpaka queue
         */
        void recordEvent(ComputeDeviceQueue const& stream);
    };
} // namespace pmacc

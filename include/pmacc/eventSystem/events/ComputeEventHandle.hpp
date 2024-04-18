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

        ComputeEventHandle(const ComputeEventHandle& other);

        /** assign an event handle
         *
         * undefined behavior if the other event handle is equal to this instance
         *
         * @param other event handle
         * @return this handle
         */
        ComputeEventHandle& operator=(const ComputeEventHandle& other);

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

/* Copyright 2013-2023 Felix Schmitt, Rene Widera
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

#include "pmacc/eventSystem/events/CudaEventHandle.hpp"
#include "pmacc/eventSystem/streams/EventStream.hpp"
#include "pmacc/eventSystem/tasks/ITask.hpp"

namespace pmacc
{
    class EventStream;

    /** Abstract base class for all tasks which depend on alpaka queue.
     */
    class StreamTask : public ITask
    {
    public:
        StreamTask();

        /**
         * Destructor.
         */
        ~StreamTask() override = default;

        /** Returns the alpaka event associated with this task.
         *
         * An event has to be recorded or set before calling this.
         *
         * @return the task's alpaka event
         */
        CudaEventHandle getCudaEventHandle() const;

        /** Sets the
         *
         * @param alpakaEvent
         */
        void setCudaEventHandle(const CudaEventHandle& alpakaEvent);

        /** Returns if this task is finished.
         *
         * @return true if the task is finished, else otherwise
         */
        bool isFinished();

        /** Returns the EventStream this StreamTask is using.
         *
         * @return pointer to the EventStream
         */
        EventStream* getEventStream();

        /** Sets the EventStream for this StreamTask.
         *
         * @param newStream new event stream
         */
        void setEventStream(EventStream* newStream);

        /** Returns the alpaka queue of the underlying EventStream.
         *
         * @return the associated alpaka queue
         */
        ComputeQueue getCudaStream();


    protected:
        /** Activates this task by recording an event on its stream.
         */
        void activate();


        EventStream* stream{nullptr};
        CudaEventHandle m_alpakaEvent;
        bool hasCudaEventHandle{false};
        bool alwaysFinished{false};
    };

} // namespace pmacc

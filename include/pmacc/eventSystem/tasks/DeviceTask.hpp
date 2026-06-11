/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/eventSystem/events/ComputeEventHandle.hpp"
#include "pmacc/eventSystem/queues/Queue.hpp"
#include "pmacc/eventSystem/tasks/ITask.hpp"

namespace pmacc
{
    class Queue;

    /** Abstract base class for all tasks which depend on alpaka queue.
     */
    class DeviceTask : public ITask
    {
    public:
        DeviceTask();

        /**
         * Destructor.
         */
        ~DeviceTask() override = default;

        /** Returns the alpaka event associated with this task.
         *
         * An event has to be recorded or set before calling this.
         *
         * @return the task's alpaka event
         */
        ComputeEventHandle getComputeEventHandle() const;

        /** Sets the
         *
         * @param alpakaEvent
         */
        void setComputeEventHandle(ComputeEventHandle const& alpakaEvent);

        /** Returns if this task is finished.
         *
         * @return true if the task is finished, else otherwise
         */
        bool isFinished();

        /** Returns the Queue this DeviceTasks is using.
         *
         * @return pointer to the Queue
         */
        Queue* getComputeDeviceQueue();

        /** Sets the Queue for this DeviceTasks.
         *
         * @param newStream new event stream
         */
        void setQueue(Queue* newStream);

        /** Returns the alpaka queue of the underlying Queue.
         *
         * @return the associated alpaka queue
         */
        ComputeDeviceQueue getAlpakaQueue();


    protected:
        /** Activates this task by recording an event on its stream.
         */
        void activate();


        Queue* stream{nullptr};
        ComputeEventHandle m_alpakaEvent;
        bool hasComputeEventHandle{false};
        bool alwaysFinished{false};
    };

} // namespace pmacc

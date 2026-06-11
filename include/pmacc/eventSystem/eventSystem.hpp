/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/eventSystem/events/EventTask.hpp"
#include "pmacc/eventSystem/mpiBarrier.hpp"
#include "pmacc/eventSystem/queues/Queue.hpp"
#include "pmacc/eventSystem/tasks/ITask.hpp"
#include "pmacc/eventSystem/waitForAllTasks.hpp"

namespace pmacc::eventSystem
{
    /**
     * Adds a new transaction to the stack.
     *
     * @param serialEvent initial base event for new transaction
     */
    void startTransaction(EventTask serialEvent = EventTask());

    /**
     * Removes the top-most transaction from the stack.
     *
     * @return the base event of the removed transaction
     */
    EventTask endTransaction();

    /**
     * Synchronizes a blocking operation with events on the top-most transaction.
     *
     * @param op operation type for synchronization
     * @return an ComputeDeviceQueue which can be used for DeviceTasks
     */
    void startOperation(ITask::TaskType op);

    /**
     * Adds event to the base event of the top-most transaction.
     *
     * @param event event to add to base event
     * @return new base event
     */
    EventTask setTransactionEvent(EventTask const& event);

    /**
     * Returns the base event of the top-most transaction.
     *
     * @return base event
     */
    EventTask getTransactionEvent();

    /** get a `Queue` that must be used for compute tasks
     *
     * depended on the opType this method is blocking
     *
     * @param opType place were the operation is running
     *               possible places are: `ITask::TASK_DEVICE`, `ITask::TASK_MPI`, `ITask::TASK_HOST`
     */
    Queue* getComputeDeviceQueue(ITask::TaskType op);
} // namespace pmacc::eventSystem

/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/eventSystem/tasks/ITask.hpp"

#include <mpi.h>

namespace pmacc
{
    /**
     * Abstract base class for all tasks which depend on MPI communication.
     */
    class MPITask : public ITask
    {
    public:
        /**
         * Constructor.
         * Starts a MPI operation on the transaction system.
         */
        MPITask() : ITask()
        {
            this->setTaskType(ITask::TASK_MPI);
        }

        /**
         * Destructor.
         */
        ~MPITask() override = default;

    protected:
        /**
         * Returns if the task is finished.
         *
         * @return if the task is finished.
         */
        inline bool isFinished()
        {
            return finished;
        }

        /**
         * Sets the task to be finished.
         */
        inline void setFinished()
        {
            finished = true;
        }

    private:
        bool finished{false};
    };
} // namespace pmacc

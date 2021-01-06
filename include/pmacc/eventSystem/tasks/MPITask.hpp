/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Benjamin Worpitz
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
        MPITask() : ITask(), finished(false)
        {
            this->setTaskType(ITask::TASK_MPI);
        }

        /**
         * Destructor.
         */
        virtual ~MPITask()
        {
        }

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
        bool finished;
    };
} // namespace pmacc

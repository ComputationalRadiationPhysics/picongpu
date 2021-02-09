/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz
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

#include <string>

namespace pmacc
{
    /**
     * EventTask is used for task-synchronization in the event system.
     *
     * Each task returns an EventTask which can be used to wait for this task
     * or let other tasks wait for this one.
     */
    class EventTask
    {
    public:
        /**
         * Constructor.
         *
         * @param taskId id for this task
         */
        EventTask(id_t taskId);

        constexpr EventTask(const pmacc::EventTask&) = default;

        /**
         * Constructor.
         */
        EventTask();

        virtual ~EventTask(){};

        /**
         * Returns the task id.
         *
         * @return id of this task
         */
        id_t getTaskId() const;

        /**
         * Returns if this task is finished.
         *
         * @return if the task is finished
         */
        bool isFinished();

        /**
         * Blocks until this task is finished.
         */
        void waitForFinished() const;

        /**
         * Adds two tasks (this task and other).
         *
         * Calls EventTask::operator+= internally.
         *
         * @param other EventTask to add to this task
         */
        EventTask operator+(const EventTask& other);

        /**
         * Adds two tasks (this task and other) and creates
         * a TaskLogicalAnd (if necessary) which is added to the Manager's queue.
         *
         * @param other EventTask to add to this task
         */
        EventTask& operator+=(const EventTask& other);

        /**
         * Copies attributes from other to this task.
         *
         * This task effectively becomes other.
         */
        EventTask& operator=(const EventTask& other);

        std::string toString();

    private:
        id_t taskId;
    };

} // namespace pmacc

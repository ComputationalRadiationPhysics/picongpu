/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

        constexpr EventTask(pmacc::EventTask const&) = default;

        /**
         * Constructor.
         */
        EventTask() = default;

        virtual ~EventTask() = default;

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
        EventTask operator+(EventTask const& other);

        /**
         * Adds two tasks (this task and other) and creates
         * a TaskLogicalAnd (if necessary) which is added to the Manager's queue.
         *
         * @param other EventTask to add to this task
         */
        EventTask& operator+=(EventTask const& other);

        /**
         * Copies attributes from other to this task.
         *
         * This task effectively becomes other.
         */
        EventTask& operator=(EventTask const& other) = default;

        std::string toString();

    private:
        id_t taskId{0};
    };

} // namespace pmacc

/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.def"
#include "pmacc/eventSystem/tasks/ITask.hpp"

#include <map>
#include <set>

namespace pmacc
{
    // forward declaration
    class EventTask;

    /**
     * Manages the event system by executing and waiting for tasks.
     */
    class Manager : public IEvent
    {
    public:
        using TaskMap = std::map<id_t, ITask*>;
        using TaskSet = std::set<id_t>;

        bool execute(id_t taskToWait = 0);

        void event(id_t eventId, EventType type, IEventData* data) override;


        /*! Return a ITask pointer if ITask is not finished
         * @return ITask pointer if Task is not finished else nullptr
         */
        ITask* getITaskIfNotFinished(id_t taskId) const;

        /**
         * blocks until the task with taskId is finished
         * @param taskId id of the task to wait for
         */
        void waitForFinished(id_t taskId);

        /** Blocks until func is ready.
         *
         * The functor is executed until it returns true. After each functor executing PMaccs event system is executed.
         *
         * @tparam T_Functor Type of the functor.
         * @param func Functor to execute. signature: `bool func()`
         *             The functor is not allowed to execute MPI collectives or any other blocking function.
         */
        template<typename T_Functor>
        void waitFor(T_Functor&& func)
        {
            while(!func())
            {
                this->execute();
            }
        }

        /**
         * blocks until all tasks in the manager are finished
         */
        void waitForAllTasks();

        /**
         * adds an ITask to the manager and returns an EventTask for it
         * @param task task to add to the manager
         */
        void addTask(ITask* task);

        /** Add a task without any dependencies
         *
         * The task is running in parallel to any other task and is never blocking the event system.
         * waitForAllTasks() will **NOT** wait until cooperative tasks are finished.
         *
         * @param task task to add to the manager
         */
        void addCooperativeTask(ITask* task);

        void addPassiveTask(ITask* task);


        std::size_t getCount();

        Manager(Manager const& cc) = delete;

        static Manager& getInstance()
        {
            static Manager instance;
            return instance;
        }

    private:
        friend struct detail::Environment;

        ITask* getPassiveITaskIfNotFinished(id_t taskId) const;

        ITask* getActiveITaskIfNotFinished(id_t taskId) const;

        Manager() = default;
        ~Manager() override;

        TaskMap tasks;
        TaskMap passiveTasks;
        TaskMap cooperativeTasks;
    };

} // namespace pmacc

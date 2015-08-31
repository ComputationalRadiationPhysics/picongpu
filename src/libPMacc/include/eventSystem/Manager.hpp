/**
 * Copyright 2013-2015 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz, Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "eventSystem/tasks/ITask.hpp"
#include "Environment.def"

#include <map>
#include <set>

namespace PMacc
{
    // forward declaration
    class EventTask;
    class EventPool;

    /**
     * Manages the event system by executing and waiting for tasks.
     */
    class Manager : public IEvent
    {
    public:
        typedef std::map<id_t, ITask*> TaskMap;
        typedef std::set<id_t> TaskSet;

        bool execute(id_t taskToWait = 0);

        void event(id_t eventId, EventType type, IEventData* data);


        /*! Return a ITask pointer if ITask is not finished
         * @return ITask pointer if Task is not finished else NULL
         */
        inline ITask* getITaskIfNotFinished(id_t taskId) const;

        /**
         * blocks until the task with taskId is finished
         * @param taskId id of the task to wait for
         */
        void waitForFinished(id_t taskId);

        /**
         * blocks until all tasks in the manager are finished
         */
        void waitForAllTasks();

        /**
         * adds an ITask to the manager and returns an EventTask for it
         * @param task task to add to the manager
         */
        void addTask(ITask *task);

        void addPassiveTask(ITask *task);

        EventPool& getEventPool();

        std::size_t getCount();

    private:

        friend class Environment<DIM1>;
        friend class Environment<DIM2>;
        friend class Environment<DIM3>;

        inline ITask* getPassiveITaskIfNotFinished(id_t taskId) const;

        inline ITask* getActiveITaskIfNotFinished(id_t taskId) const;

        Manager();

        Manager(const Manager& cc);

        virtual ~Manager();

        static Manager& getInstance()
        {
            static Manager instance;
            return instance;
        }

        TaskMap tasks;
        TaskMap passiveTasks;
        EventPool *eventPool;
    };

} //namespace PMacc

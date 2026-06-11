/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pmacc/eventSystem/Manager.hpp"

#include "pmacc/assert.hpp"
#include "pmacc/eventSystem/queues/QueueController.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <set>

// #define DEBUG_EVENTS

namespace pmacc
{
    Manager::~Manager()
    {
        waitForAllTasks();
        PMACC_ASSERT(cooperativeTasks.size() == 0u);
    }

    bool Manager::execute(id_t taskToWait)
    {
#ifdef DEBUG_EVENTS
        static int old_max = 0;
        static int deep = -1;
        static int counter = 0;
        ++counter;

        deep++;
        if(deep > old_max)
        {
            old_max = deep;
        }
#endif

        static auto iterCooperative = cooperativeTasks.begin();
        if(iterCooperative == cooperativeTasks.end())
            iterCooperative = cooperativeTasks.begin();
        while(iterCooperative != cooperativeTasks.end())
        {
            id_t id = iterCooperative->first;
            ITask* taskPtr = iterCooperative->second;
            PMACC_ASSERT(taskPtr != nullptr);
            ++iterCooperative;
            if(taskPtr->execute())
            {
                auto it = cooperativeTasks.find(id);
                if(it != cooperativeTasks.end())
                {
                    cooperativeTasks.erase(id);
                    delete taskPtr;
                }
            }
        }

        static auto iter = tasks.begin();

        if(iter == tasks.end())
            iter = tasks.begin();

        // this is the slow but very save variant to delete tasks in a map
        while(iter != tasks.end())
        {
            id_t id = iter->first;
            ITask* taskPtr = iter->second;
            PMACC_ASSERT(taskPtr != nullptr);
            ++iter;
#ifdef DEBUG_EVENTS
            if(counter == 500000)
                std::cout << taskPtr->toString() << " " << passiveTasks.size() << std::endl;
#endif
            if(taskPtr->execute())
            {
                /*test if task is deleted by other stackdeep*/
                if(getActiveITaskIfNotFinished(id) == taskPtr)
                {
                    tasks.erase(id);
                    delete taskPtr;
                }
#ifdef DEBUG_EVENTS
                counter = 0;
#endif

                if(taskToWait == id)
                {
                    iter = tasks.end();
#ifdef DEBUG_EVENTS
                    --deep;
#endif
                    return true; // jump out because searched task is finished
                }
            }
        }

#ifdef DEBUG_EVENTS
        --deep;
#endif

        return false;
    }

    void Manager::event(id_t eventId, EventType, IEventData*)
    {
        passiveTasks.erase(eventId);
    }

    ITask* Manager::getITaskIfNotFinished(id_t taskId) const
    {
        if(taskId == 0)
            return nullptr;
        ITask* passiveTask = getPassiveITaskIfNotFinished(taskId);
        if(passiveTask != nullptr)
            return passiveTask;

        return getActiveITaskIfNotFinished(taskId);
    }

    ITask* Manager::getPassiveITaskIfNotFinished(id_t taskId) const
    {
        auto itPassive = passiveTasks.find(taskId);
        if(itPassive != passiveTasks.end())
            return itPassive->second;
        return nullptr;
    }

    ITask* Manager::getActiveITaskIfNotFinished(id_t taskId) const
    {
        auto it = tasks.find(taskId);
        if(it != tasks.end())
            return it->second;
        return nullptr;
    }

    void Manager::waitForFinished(id_t taskId)
    {
        if(taskId == 0)
            return;
        // check if task is passive and wait on it
        ITask* task = getPassiveITaskIfNotFinished(taskId);
        if(task != nullptr)
        {
            do
            {
                this->execute();
            } while(getPassiveITaskIfNotFinished(taskId) != nullptr);

            return; // we can jump out because task is passive task
        }

        // check if task is  active and wait on it
        task = getActiveITaskIfNotFinished(taskId);
        if(task != nullptr)
        {
            do
            {
                if(this->execute(taskId))
                    return; // jump out because task is finished
            } while(getActiveITaskIfNotFinished(taskId) != nullptr);
        }
    }

    void Manager::waitForAllTasks()
    {
        while(tasks.size() != 0 || passiveTasks.size() != 0)
        {
            this->execute();
        }
        PMACC_ASSERT(tasks.size() == 0);
    }

    void Manager::addTask(ITask* task)
    {
        PMACC_ASSERT(task != nullptr);
        tasks[task->getId()] = task;
    }

    void Manager::addCooperativeTask(ITask* task)
    {
        PMACC_ASSERT(task != nullptr);
        cooperativeTasks[task->getId()] = task;
    }

    void Manager::addPassiveTask(ITask* task)
    {
        PMACC_ASSERT(task != nullptr);

        task->addObserver(this);
        passiveTasks[task->getId()] = task;
    }

    std::size_t Manager::getCount()
    {
        for(auto iter = tasks.begin(); iter != tasks.end(); ++iter)
        {
            if(iter->second != nullptr)
            {
                std::cout << iter->first << " = " << iter->second->toString() << std::endl;
            }
        }
        return tasks.size();
    }

} // namespace pmacc

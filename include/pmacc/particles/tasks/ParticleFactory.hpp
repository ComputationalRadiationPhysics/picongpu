/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/Environment.def"

namespace pmacc
{
    /**
     * Singleton Factory-pattern class for creation of several types of EventTasks.
     * Tasks are not actually 'returned' but immediately initialised and
     * added to the Manager's queue. An exception is TaskKernel.
     */
    class ParticleFactory
    {
    public:
        /**
         * Creates a TaskReceive.
         * @param ex Exchange to create new TaskReceive with
         * @param task_out returns the newly created task
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an
         * observer
         */
        template<class ParBase>
        EventTask createTaskParticlesReceive(ParBase& parBuffer, ITask* registeringTask = nullptr);

        template<class ParBase>
        EventTask createTaskReceiveParticlesExchange(
            ParBase& parBase,
            uint32_t exchange,
            ITask* registeringTask = nullptr);

        /**
         * Creates a TaskSend.
         * @param ex Exchange to create new TaskSend with
         * @param task_in TaskReceive to register at new TaskSend
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an
         * observer
         */
        template<class ParBase>
        EventTask createTaskParticlesSend(ParBase& parBase, ITask* registeringTask = nullptr);

        template<class ParBase>
        EventTask createTaskSendParticlesExchange(
            ParBase& parBase,
            uint32_t exchange,
            ITask* registeringTask = nullptr);


    private:
        friend struct detail::Environment;

        /**
         * returns the instance of this factory
         * @return the instance
         */
        static ParticleFactory& getInstance()
        {
            static ParticleFactory instance;
            return instance;
        }

        ParticleFactory() = default;

        ParticleFactory(ParticleFactory const&) = default;
    };

} // namespace pmacc

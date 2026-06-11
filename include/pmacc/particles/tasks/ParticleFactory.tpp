/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/particles/tasks/ParticleFactory.hpp"
#include "pmacc/particles/tasks/TaskParticlesReceive.hpp"
#include "pmacc/particles/tasks/TaskParticlesSend.hpp"
#include "pmacc/particles/tasks/TaskReceiveParticlesExchange.hpp"
#include "pmacc/particles/tasks/TaskSendParticlesExchange.hpp"

namespace pmacc
{
    template<class ParBase>
    inline EventTask ParticleFactory::createTaskParticlesReceive(ParBase& parBase, ITask* registeringTask)
    {
        auto* task = new TaskParticlesReceive<ParBase>(parBase);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }

    template<class ParBase>
    inline EventTask ParticleFactory::createTaskReceiveParticlesExchange(
        ParBase& parBase,
        uint32_t exchange,
        ITask* registeringTask)
    {
        auto* task = new TaskReceiveParticlesExchange<ParBase>(parBase, exchange);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }

    template<class ParBase>
    inline EventTask ParticleFactory::createTaskParticlesSend(ParBase& parBase, ITask* registeringTask)
    {
        auto* task = new TaskParticlesSend<ParBase>(parBase);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }

    template<class ParBase>
    inline EventTask ParticleFactory::createTaskSendParticlesExchange(
        ParBase& parBase,
        uint32_t exchange,
        ITask* registeringTask)
    {
        auto* task = new TaskSendParticlesExchange<ParBase>(parBase, exchange);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }


} // namespace pmacc

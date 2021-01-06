/* Copyright 2013-2021 Rene Widera
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

#include "pmacc/Environment.hpp"
#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/particles/tasks/ParticleFactory.hpp"

#include "pmacc/particles/tasks/TaskSendParticlesExchange.hpp"
#include "pmacc/particles/tasks/TaskReceiveParticlesExchange.hpp"
#include "pmacc/particles/tasks/TaskParticlesReceive.hpp"
#include "pmacc/particles/tasks/TaskParticlesSend.hpp"

namespace pmacc
{
    template<class ParBase>
    inline EventTask ParticleFactory::createTaskParticlesReceive(ParBase& parBase, ITask* registeringTask)
    {
        TaskParticlesReceive<ParBase>* task = new TaskParticlesReceive<ParBase>(parBase);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }

    template<class ParBase>
    inline EventTask ParticleFactory::createTaskReceiveParticlesExchange(
        ParBase& parBase,
        uint32_t exchange,
        ITask* registeringTask)
    {
        TaskReceiveParticlesExchange<ParBase>* task = new TaskReceiveParticlesExchange<ParBase>(parBase, exchange);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }

    template<class ParBase>
    inline EventTask ParticleFactory::createTaskParticlesSend(ParBase& parBase, ITask* registeringTask)
    {
        TaskParticlesSend<ParBase>* task = new TaskParticlesSend<ParBase>(parBase);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }

    template<class ParBase>
    inline EventTask ParticleFactory::createTaskSendParticlesExchange(
        ParBase& parBase,
        uint32_t exchange,
        ITask* registeringTask)
    {
        TaskSendParticlesExchange<ParBase>* task = new TaskSendParticlesExchange<ParBase>(parBase, exchange);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }


} // namespace pmacc

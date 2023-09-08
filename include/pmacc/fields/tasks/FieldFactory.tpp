/* Copyright 2013-2023 Rene Widera
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

#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/eventSystem/tasks/ITask.hpp"
#include "pmacc/fields/tasks/FieldFactory.hpp"
#include "pmacc/fields/tasks/TaskFieldReceiveAndInsert.hpp"
#include "pmacc/fields/tasks/TaskFieldReceiveAndInsertExchange.hpp"
#include "pmacc/fields/tasks/TaskFieldSend.hpp"
#include "pmacc/fields/tasks/TaskFieldSendExchange.hpp"

namespace pmacc
{
    template<class Field>
    inline EventTask FieldFactory::createTaskFieldReceiveAndInsert(Field& buffer, ITask* registeringTask)
    {
        auto* task = new TaskFieldReceiveAndInsert<Field>(buffer);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }

    template<class Field>
    inline EventTask FieldFactory::createTaskFieldReceiveAndInsertExchange(
        Field& buffer,
        uint32_t exchange,
        ITask* registeringTask)
    {
        auto* task = new TaskFieldReceiveAndInsertExchange<Field>(buffer, exchange);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }

    template<class Field>
    inline EventTask FieldFactory::createTaskFieldSend(Field& buffer, ITask* registeringTask)
    {
        auto* task = new TaskFieldSend<Field>(buffer);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }

    template<class Field>
    inline EventTask FieldFactory::createTaskFieldSendExchange(
        Field& buffer,
        uint32_t exchange,
        ITask* registeringTask)
    {
        auto* task = new TaskFieldSendExchange<Field>(buffer, exchange);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }


} // namespace pmacc

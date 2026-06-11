/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

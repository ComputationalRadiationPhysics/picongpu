/**
 * Copyright 2013 Rene Widera
 *
 * This file is part of PIConGPU. 
 * 
 * PIConGPU is free software: you can redistribute it and/or modify 
 * it under the terms of the GNU General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * 
 * PIConGPU is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * along with PIConGPU.  
 * If not, see <http://www.gnu.org/licenses/>. 
 */ 
 

#include "fields/tasks/FieldFactory.hpp"
#include "fields/tasks/TaskFieldReceiveAndInsert.hpp"
#include "fields/tasks/TaskFieldReceiveAndInsertExchange.hpp"
#include "fields/tasks/TaskFieldSend.hpp"
#include "fields/tasks/TaskFieldSendExchange.hpp"

#include "eventSystem/tasks/Factory.hpp"
#include "eventSystem/tasks/ITask.hpp"

namespace PMacc
{

    template<class Field>
    inline EventTask FieldFactory::createTaskFieldReceiveAndInsert(Field &buffer,
                                                                   ITask *registeringTask)
    {
        TaskFieldReceiveAndInsert<Field>* task = new TaskFieldReceiveAndInsert<Field > (buffer);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }

    template<class Field>
    inline EventTask FieldFactory::createTaskFieldReceiveAndInsertExchange(Field &buffer, uint32_t exchange,
                                                                           ITask *registeringTask)
    {
        TaskFieldReceiveAndInsertExchange<Field>* task = new TaskFieldReceiveAndInsertExchange<Field > (buffer, exchange);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }

    template<class Field>
    inline EventTask FieldFactory::createTaskFieldSend(Field &buffer,
                                                       ITask *registeringTask)
    {
        TaskFieldSend<Field>* task = new TaskFieldSend<Field > (buffer);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }

    template<class Field>
    inline EventTask FieldFactory::createTaskFieldSendExchange(Field &buffer, uint32_t exchange,
                                                               ITask *registeringTask)
    {
        TaskFieldSendExchange<Field>* task = new TaskFieldSendExchange<Field > (buffer, exchange);

        return Environment<>::get().Factory().startTask(*task, registeringTask);
    }



} //namespace PMacc





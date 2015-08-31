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

#pragma once

#include "eventSystem/EventSystem.hpp"
#include "fields/tasks/FieldFactory.hpp"
#include "eventSystem/tasks/ITask.hpp"
#include "eventSystem/tasks/MPITask.hpp"
#include "eventSystem/events/EventDataReceive.hpp"



namespace PMacc
{

template<class Field>
class TaskFieldReceiveAndInsertExchange : public MPITask
{
public:

    TaskFieldReceiveAndInsertExchange(Field &buffer, uint32_t exchange) :
    m_buffer(buffer),
    m_exchange(exchange),
    m_state(Constructor),
    initDependency(__getTransactionEvent())
    {
    }

    virtual void init()
    {
        m_state = Init;
        initDependency = m_buffer.getGridBuffer().asyncReceive(initDependency, m_exchange);
        m_state = WaitForReceive;
    }

    bool executeIntern()
    {
        switch (m_state)
        {
        case Init:
            break;
        case WaitForReceive:
            if (NULL == Environment<>::get().Manager().getITaskIfNotFinished(initDependency.getTaskId()))
            {
                m_state = Finished;
                return true;
            }
            break;
        case Finished:
            return true;
        default:
            return false;
        }

        return false;
    }

    virtual ~TaskFieldReceiveAndInsertExchange()
    {
        notify(this->myId, RECVFINISHED, NULL);
    }

    void event(id_t, EventType, IEventData*)
    {
    }

    std::string toString()
    {
        std::ostringstream stateNumber;
        stateNumber << m_state;
        return std::string("TaskFieldReceiveAndInsertExchange/") + stateNumber.str();
    }

private:

    enum state_t
    {
        Constructor,
        Init,
        WaitForReceive,
        Finished

    };




    Field& m_buffer;
    state_t m_state;
    EventTask insertEvent;
    EventTask initDependency;
    uint32_t m_exchange;
};

} //namespace PMacc



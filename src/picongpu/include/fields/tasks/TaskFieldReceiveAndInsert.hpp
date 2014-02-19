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



#ifndef _TASKFIELDRECEIVEANDINSERT_HPP
#define	_TASKFIELDRECEIVEANDINSERT_HPP

#include "simulation_defines.hpp"
#include "eventSystem/EventSystem.hpp"
#include "fields/tasks/FieldFactory.hpp"
#include "eventSystem/tasks/ITask.hpp"
#include "eventSystem/tasks/MPITask.hpp"
#include "eventSystem/events/EventDataReceive.hpp"
#include "eventSystem/EventSystem.hpp"
#include <iostream>


namespace PMacc
{

template<class Field>
class TaskFieldReceiveAndInsert : public MPITask
{
public:


    static const uint32_t Dim = picongpu::simDim;
 
    TaskFieldReceiveAndInsert(Field &buffer) :
    buffer(buffer),
    state(Constructor)
    {
    }

    virtual void init()
    {
        state = Init;
        EventTask serialEvent = __getTransactionEvent();

        for (uint32_t i = 1; i < numberOfNeighbors[Dim]; ++i)
        {
            if (buffer.getGridBuffer().hasReceiveExchange(i))
            {
                __startAtomicTransaction(serialEvent);
                FieldFactory::getInstance().createTaskFieldReceiveAndInsertExchange(buffer, i);
                tmpEvent += __endTransaction();
            }
        }
        state = WaitForReceived;
    }

    bool executeIntern()
    {
        switch (state)
        {
        case Init:
            break;
        case WaitForReceived:
            if (NULL == Manager::getInstance().getITaskIfNotFinished(tmpEvent.getTaskId()))
            {
                state = Insert;
            }
            break;
        case Insert:
            state = Wait;
            __startAtomicTransaction();
            for (uint32_t i = 1; i < numberOfNeighbors[Dim]; ++i)
            {
                if (buffer.getGridBuffer().hasReceiveExchange(i))
                {
                    buffer.insertField(i);
                }
            }
            tmpEvent = __endTransaction();
            state = WaitInsertFinished;
            break;
        case Wait:
            break;
        case WaitInsertFinished:
            if (NULL == Manager::getInstance().getITaskIfNotFinished(tmpEvent.getTaskId()))
            {
                state = Finish;
                return true;
            }
            break;
        case Finish:
            return true;
        default:
            return false;
        }

        return false;
    }

    virtual ~TaskFieldReceiveAndInsert()
    {
        notify(this->myId, RECVFINISHED, NULL);
    }

    void event(id_t, EventType, IEventData*)
    {
    }

    std::string toString()
    {
        return "TaskFieldReceiveAndInsert";
    }

private:

    enum state_t
    {
        Constructor,
        Init,
        Wait,
        Insert,
        WaitInsertFinished,
        WaitForReceived,
        Finish

    };


    Field& buffer;
    state_t state;
    EventTask tmpEvent;

};

} //namespace PMacc


#endif	/* _TASKFIELDRECEIVEANDINSERT_HPP */


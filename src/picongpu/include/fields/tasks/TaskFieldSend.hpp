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
 


#ifndef _TASKFIELDSEND_HPP
#define	_TASKFIELDSEND_HPP


#include "eventSystem/EventSystem.hpp"
#include "fields/tasks/FieldFactory.hpp"
#include "eventSystem/tasks/ITask.hpp"
#include "eventSystem/tasks/MPITask.hpp"
#include "eventSystem/events/EventDataReceive.hpp"


namespace PMacc
{

    template<class Field>
    class TaskFieldSend : public MPITask
    {
    public:

        enum
        {
            Dim = picongpu::simDim
        };

        TaskFieldSend(Field &buffer) :
        buffer(buffer),
        state(Constructor) { }

        virtual void init()
        {
            state = Init;
            EventTask serialEvent = __getTransactionEvent();

            for (uint32_t i = 1; i < numberOfNeighbors[Dim]; ++i)
            {
                if (buffer.getGridBuffer().hasSendExchange(i))
                {
                    __startAtomicTransaction(serialEvent);
                    FieldFactory::getInstance().createTaskFieldSendExchange(buffer, i);
                    tmpEvent += __endTransaction();
                }
            }
            state = WaitForSend;
        }

        bool executeIntern()
        {
            switch (state)
            {
                case Init:
                    break;
                case WaitForSend:
                    return NULL == Manager::getInstance().getITaskIfNotFinished(tmpEvent.getTaskId());
                default:
                    return false;
            }

            return false;
        }

        virtual ~TaskFieldSend()
        {
            notify(this->myId, SENDFINISHED, NULL);
        }

        void event(id_t, EventType, IEventData*) { }

        std::string toString()
        {
            return "TaskFieldSend";
        }

    private:

        enum state_t
        {
            Constructor,
            Init,
            WaitForSend

        };


        Field& buffer;
        state_t state;
        EventTask tmpEvent;
    };

} //namespace PMacc


#endif	/* _TASKFIELDSEND_HPP */


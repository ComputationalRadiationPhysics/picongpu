/**
 * Copyright 2013 René Widera
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
 


#ifndef  _TASKFIELDSENDEXCHANGE_HPP
#define	_TASKFIELDSENDEXCHANGE_HPP


#include "eventSystem/EventSystem.hpp"
#include "fields/tasks/FieldFactory.hpp"
#include "eventSystem/tasks/ITask.hpp"
#include "eventSystem/tasks/MPITask.hpp"
#include "eventSystem/events/EventDataReceive.hpp"



namespace PMacc
{

    template<class Field>
    class TaskFieldSendExchange : public MPITask
    {
    public:

        enum
        {
            Dim = DIM3,
            /* Exchanges in 2D=9 and in 3D=27
             */
            Exchanges = 27
        };

        TaskFieldSendExchange(Field &buffer, uint32_t exchange) :
        buffer(buffer),
        exchange(exchange),
        state(Constructor),
        initDependency(__getTransactionEvent())
        {
        }

        virtual void init()
        {
            state = Init;
            __startTransaction(initDependency);
            buffer.bashField(exchange);
            initDependency = __endTransaction();
            state = WaitForBash;
        }

        bool executeIntern()
        {
            switch (state)
            {
            case Init:
                break;
            case WaitForBash:

                if (NULL == Manager::getInstance().getITaskIfNotFinished(initDependency.getTaskId()) )
                {
                    state = InitSend;
                    sendEvent = buffer.getGridBuffer().asyncSend(EventTask(), exchange, initDependency);
                    state = WaitForSendEnd;
                }

                break;
            case InitSend:
                break;
            case WaitForSendEnd:
                if (NULL == Manager::getInstance().getITaskIfNotFinished(sendEvent.getTaskId()))
                {
                    state = Finished;
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

        virtual ~TaskFieldSendExchange()
        {
            notify(this->myId, SENDFINISHED, NULL);
        }

        void event(id_t eventId, EventType type, IEventData* data)
        {
        }

        std::string toString()
        {
            return "TaskFieldSendExchange";
        }

    private:

        enum state_t
        {
            Constructor,
            Init,
            WaitForBash,
            InitSend,
            WaitForSendEnd,
            Finished

        };


        Field& buffer;
        state_t state;
        EventTask sendEvent;
        EventTask initDependency;
        uint32_t exchange;
    };

} //namespace PMacc


#endif	/* _TASKFIELDSENDEXCHANGE_HPP */


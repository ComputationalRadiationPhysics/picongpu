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

#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/fields/tasks/FieldFactory.hpp"
#include "pmacc/eventSystem/tasks/ITask.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"
#include "pmacc/eventSystem/events/EventDataReceive.hpp"
#include "pmacc/eventSystem/EventSystem.hpp"
#include <iostream>
#include "pmacc/traits/NumberOfExchanges.hpp"

namespace pmacc
{
    template<class Field>
    class TaskFieldReceiveAndInsert : public MPITask
    {
    public:
        static constexpr uint32_t Dim = picongpu::simDim;

        TaskFieldReceiveAndInsert(Field& buffer) : m_buffer(buffer), m_state(Constructor)
        {
        }

        virtual void init()
        {
            m_state = Init;
            EventTask serialEvent = __getTransactionEvent();

            for(uint32_t i = 1; i < traits::NumberOfExchanges<Dim>::value; ++i)
            {
                if(m_buffer.getGridBuffer().hasReceiveExchange(i))
                {
                    __startTransaction(serialEvent);
                    FieldFactory::getInstance().createTaskFieldReceiveAndInsertExchange(m_buffer, i);
                    m_tmpEvent += __endTransaction();
                }
            }
            m_state = WaitForReceived;
        }

        bool executeIntern()
        {
            switch(m_state)
            {
            case Init:
                break;
            case WaitForReceived:
                if(nullptr == Environment<>::get().Manager().getITaskIfNotFinished(m_tmpEvent.getTaskId()))
                {
                    m_state = Insert;
                }
                break;
            case Insert:
                m_state = Wait;
                __startTransaction();
                for(uint32_t i = 1; i < traits::NumberOfExchanges<Dim>::value; ++i)
                {
                    if(m_buffer.getGridBuffer().hasReceiveExchange(i))
                    {
                        m_buffer.insertField(i);
                    }
                }
                m_tmpEvent = __endTransaction();
                m_state = WaitInsertFinished;
                break;
            case Wait:
                break;
            case WaitInsertFinished:
                if(nullptr == Environment<>::get().Manager().getITaskIfNotFinished(m_tmpEvent.getTaskId()))
                {
                    m_state = Finish;
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
            notify(this->myId, RECVFINISHED, nullptr);
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


        Field& m_buffer;
        state_t m_state;
        EventTask m_tmpEvent;
    };

} // namespace pmacc

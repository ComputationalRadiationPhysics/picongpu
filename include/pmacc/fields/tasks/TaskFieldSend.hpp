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
#include "pmacc/traits/NumberOfExchanges.hpp"

namespace pmacc
{
    template<class Field>
    class TaskFieldSend : public MPITask
    {
    public:
        enum
        {
            Dim = picongpu::simDim
        };

        TaskFieldSend(Field& buffer) : m_buffer(buffer), m_state(Constructor)
        {
        }

        virtual void init()
        {
            m_state = Init;
            EventTask serialEvent = __getTransactionEvent();

            for(uint32_t i = 1; i < traits::NumberOfExchanges<Dim>::value; ++i)
            {
                if(m_buffer.getGridBuffer().hasSendExchange(i))
                {
                    __startTransaction(serialEvent);
                    FieldFactory::getInstance().createTaskFieldSendExchange(m_buffer, i);
                    tmpEvent += __endTransaction();
                }
            }
            m_state = WaitForSend;
        }

        bool executeIntern()
        {
            switch(m_state)
            {
            case Init:
                break;
            case WaitForSend:
                return nullptr == Environment<>::get().Manager().getITaskIfNotFinished(tmpEvent.getTaskId());
            default:
                return false;
            }

            return false;
        }

        virtual ~TaskFieldSend()
        {
            notify(this->myId, SENDFINISHED, nullptr);
        }

        void event(id_t, EventType, IEventData*)
        {
        }

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


        Field& m_buffer;
        state_t m_state;
        EventTask tmpEvent;
    };

} // namespace pmacc

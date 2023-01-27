/* Copyright 2013-2022 Rene Widera
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


#include "pmacc/eventSystem/events/EventDataReceive.hpp"
#include "pmacc/eventSystem/tasks/ITask.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"
#include "pmacc/fields/tasks/FieldFactory.hpp"


namespace pmacc
{
    template<class Field>
    class TaskFieldSendExchange : public MPITask
    {
    public:
        TaskFieldSendExchange(Field& buffer, uint32_t exchange)
            : m_buffer(buffer)
            , m_state(Constructor)
            , m_initDependency(eventSystem::getTransactionEvent())
            , m_exchange(exchange)
        {
        }

        void init() override
        {
            m_state = Init;
            eventSystem::startTransaction(m_initDependency);
            m_buffer.bashField(m_exchange);
            m_initDependency = eventSystem::endTransaction();
            m_state = WaitForBash;
        }

        bool executeIntern() override
        {
            switch(m_state)
            {
            case Init:
                break;
            case WaitForBash:

                if(nullptr == Manager::getInstance().getITaskIfNotFinished(m_initDependency.getTaskId()))
                {
                    m_state = InitSend;
                    m_sendEvent = m_buffer.getGridBuffer().asyncSend(EventTask(), m_exchange);
                    m_initDependency = m_sendEvent;
                    m_state = WaitForSendEnd;
                }

                break;
            case InitSend:
                break;
            case WaitForSendEnd:
                if(nullptr == Manager::getInstance().getITaskIfNotFinished(m_sendEvent.getTaskId()))
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

        ~TaskFieldSendExchange() override
        {
            notify(this->myId, SENDFINISHED, nullptr);
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
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


        Field& m_buffer;
        state_t m_state;
        EventTask m_sendEvent;
        EventTask m_initDependency;
        uint32_t m_exchange;
    };

} // namespace pmacc

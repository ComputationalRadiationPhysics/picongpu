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


#include "pmacc/eventSystem/Manager.hpp"
#include "pmacc/eventSystem/events/EventDataReceive.hpp"
#include "pmacc/eventSystem/tasks/ITask.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"
#include "pmacc/fields/tasks/FieldFactory.hpp"


namespace pmacc
{
    template<class Field>
    class TaskFieldReceiveAndInsertExchange : public MPITask
    {
    public:
        TaskFieldReceiveAndInsertExchange(Field& buffer, uint32_t exchange)
            : m_buffer(buffer)
            , m_state(Constructor)
            , initDependency(eventSystem::getTransactionEvent())
            , m_exchange(exchange)
        {
        }

        void init() override
        {
            m_state = Init;
            initDependency = m_buffer.getGridBuffer().asyncReceive(initDependency, m_exchange);
            m_state = WaitForReceive;
        }

        bool executeIntern() override
        {
            switch(m_state)
            {
            case Init:
                break;
            case WaitForReceive:
                if(nullptr == Manager::getInstance().getITaskIfNotFinished(initDependency.getTaskId()))
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

        ~TaskFieldReceiveAndInsertExchange() override
        {
            notify(this->myId, RECVFINISHED, nullptr);
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
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

} // namespace pmacc

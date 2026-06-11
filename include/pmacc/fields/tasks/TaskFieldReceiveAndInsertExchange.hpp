/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

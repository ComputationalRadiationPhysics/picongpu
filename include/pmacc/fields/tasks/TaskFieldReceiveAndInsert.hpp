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
#include "pmacc/traits/NumberOfExchanges.hpp"

#include <cstdint>
#include <iostream>

namespace pmacc
{
    template<class Field>
    class TaskFieldReceiveAndInsert : public MPITask
    {
    public:
        static constexpr uint32_t Dim = Field::dim;

        TaskFieldReceiveAndInsert(Field& buffer) : m_buffer(buffer), m_state(Constructor)
        {
        }

        void init() override
        {
            m_state = Init;
            EventTask serialEvent = eventSystem::getTransactionEvent();

            for(uint32_t i = 1; i < traits::NumberOfExchanges<Dim>::value; ++i)
            {
                if(m_buffer.getGridBuffer().hasReceiveExchange(i))
                {
                    eventSystem::startTransaction(serialEvent);
                    FieldFactory::getInstance().createTaskFieldReceiveAndInsertExchange(m_buffer, i);
                    m_tmpEvent += eventSystem::endTransaction();
                }
            }
            m_state = WaitForReceived;
        }

        bool executeIntern() override
        {
            switch(m_state)
            {
            case Init:
                break;
            case WaitForReceived:
                if(nullptr == Manager::getInstance().getITaskIfNotFinished(m_tmpEvent.getTaskId()))
                {
                    m_state = Insert;
                }
                break;
            case Insert:
                m_state = Wait;
                eventSystem::startTransaction();
                for(uint32_t i = 1; i < traits::NumberOfExchanges<Dim>::value; ++i)
                {
                    if(m_buffer.getGridBuffer().hasReceiveExchange(i))
                    {
                        m_buffer.insertField(i);
                    }
                }
                m_tmpEvent = eventSystem::endTransaction();
                m_state = WaitInsertFinished;
                break;
            case Wait:
                break;
            case WaitInsertFinished:
                if(nullptr == Manager::getInstance().getITaskIfNotFinished(m_tmpEvent.getTaskId()))
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

        ~TaskFieldReceiveAndInsert() override
        {
            notify(this->myId, RECVFINISHED, nullptr);
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
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

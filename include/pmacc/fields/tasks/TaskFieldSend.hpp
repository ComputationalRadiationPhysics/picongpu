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

namespace pmacc
{
    template<class Field>
    class TaskFieldSend : public MPITask
    {
    public:
        static constexpr uint32_t Dim = Field::dim;

        TaskFieldSend(Field& buffer) : m_buffer(buffer), m_state(Constructor)
        {
        }

        void init() override
        {
            m_state = Init;
            EventTask serialEvent = eventSystem::getTransactionEvent();

            for(uint32_t i = 1; i < traits::NumberOfExchanges<Dim>::value; ++i)
            {
                if(m_buffer.getGridBuffer().hasSendExchange(i))
                {
                    eventSystem::startTransaction(serialEvent);
                    FieldFactory::getInstance().createTaskFieldSendExchange(m_buffer, i);
                    tmpEvent += eventSystem::endTransaction();
                }
            }
            m_state = WaitForSend;
        }

        bool executeIntern() override
        {
            switch(m_state)
            {
            case Init:
                break;
            case WaitForSend:
                return nullptr == Manager::getInstance().getITaskIfNotFinished(tmpEvent.getTaskId());
            default:
                return false;
            }

            return false;
        }

        ~TaskFieldSend() override
        {
            notify(this->myId, SENDFINISHED, nullptr);
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
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

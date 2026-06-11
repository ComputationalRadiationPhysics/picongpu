/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/eventSystem/Manager.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"

namespace pmacc
{
    template<class ParBase>
    class TaskReceiveParticlesExchange : public MPITask
    {
    public:
        TaskReceiveParticlesExchange(ParBase& parBase, uint32_t exchange)
            : parBase(parBase)
            , state(Constructor)
            , initDependency(eventSystem::getTransactionEvent())
            , exchange(exchange)
            , maxSize(parBase.getParticlesBuffer().getReceiveExchangeStack(exchange).getMaxParticlesCount())
            , lastSize(0)
        {
        }

        void init() override
        {
            state = Init;
            lastReceiveEvent = parBase.getParticlesBuffer().asyncReceiveParticles(initDependency, exchange);
            initDependency = lastReceiveEvent;
            state = WaitForReceive;
        }

        bool executeIntern() override
        {
            switch(state)
            {
            case Init:
                break;
            case WaitForReceive:

                if(nullptr == Manager::getInstance().getITaskIfNotFinished(lastReceiveEvent.getTaskId()))
                {
                    state = InitInsert;
                    // bash is finished
                    eventSystem::startTransaction();
                    lastSize
                        = parBase.getParticlesBuffer().getReceiveExchangeStack(exchange).getHostParticlesCurrentSize();
                    parBase.insertParticles(exchange);
                    tmpEvent = eventSystem::endTransaction();
                    initDependency = tmpEvent;
                    state = WaitForInsert;
                }

                break;
            case InitInsert:
                break;
            case WaitForInsert:
                if(nullptr == Manager::getInstance().getITaskIfNotFinished(tmpEvent.getTaskId()))
                {
                    state = Wait;
                    PMACC_ASSERT(lastSize <= maxSize);
                    // check for next bash round
                    if(lastSize == maxSize)
                        init(); // call init and run a full send cycle
                    else
                    {
                        state = Finished;
                        return true;
                    }
                }
                break;
            case Wait:
                break;
            case Finished:
                return true;
            default:
                return false;
            }

            return false;
        }

        ~TaskReceiveParticlesExchange() override
        {
            notify(this->myId, RECVFINISHED, nullptr);
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
        {
            return "TaskReceiveParticlesExchange";
        }

    private:
        enum state_t
        {
            Constructor,
            Init,
            WaitForReceive,
            InitInsert,
            WaitForInsert,
            Wait,
            Finished
        };

        ParBase& parBase;
        state_t state;
        EventTask tmpEvent;
        EventTask lastReceiveEvent;
        EventTask initDependency;
        uint32_t exchange;
        size_t maxSize;
        size_t lastSize;
    };

} // namespace pmacc

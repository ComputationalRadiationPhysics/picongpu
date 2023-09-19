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

#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/eventSystem/Manager.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"
#include "pmacc/type/Exchange.hpp"

namespace pmacc
{
    template<class ParBase>
    class TaskSendParticlesExchange : public MPITask
    {
    public:
        TaskSendParticlesExchange(ParBase& parBase, uint32_t exchange)
            : parBase(parBase)
            , state(Constructor)
            , lastSendEvent(EventTask())
            , initDependency(eventSystem::getTransactionEvent())
            , exchange(exchange)
            , maxSize(parBase.getParticlesBuffer().getSendExchangeStack(exchange).getMaxParticlesCount())
            , lastSize(0)
            , retryCounter(0)
        {
        }

        void init() override
        {
            state = Init;
            eventSystem::startTransaction(initDependency);
            parBase.copyGuardToExchange(exchange);
            tmpEvent = eventSystem::endTransaction();
            state = WaitForBash;
        }

        bool executeIntern() override
        {
            switch(state)
            {
            case Init:
                break;
            case WaitForBash:

                if(nullptr == Manager::getInstance().getITaskIfNotFinished(tmpEvent.getTaskId())
                   && nullptr == Manager::getInstance().getITaskIfNotFinished(lastSendEvent.getTaskId()))
                {
                    state = InitSend;
                    // bash is finished
                    eventSystem::startTransaction();
                    lastSize
                        = parBase.getParticlesBuffer().getSendExchangeStack(exchange).getDeviceParticlesCurrentSize();
                    lastSendEvent = parBase.getParticlesBuffer().asyncSendParticles(
                        eventSystem::getTransactionEvent(),
                        exchange);
                    initDependency = lastSendEvent;
                    eventSystem::endTransaction();
                    state = WaitForSend;
                }

                break;
            case InitSend:
                break;
            case WaitForSend:
                if(nullptr == Manager::getInstance().getITaskIfNotFinished(tmpEvent.getTaskId()))
                {
                    PMACC_ASSERT(lastSize <= maxSize);
                    // check for next bash round
                    if(lastSize == maxSize)
                    {
                        ++retryCounter;
                        init(); // call init and run a full send cycle
                    }
                    else
                        state = WaitForSendEnd;
                }
                break;
            case WaitForSendEnd:
                if(nullptr == Manager::getInstance().getITaskIfNotFinished(lastSendEvent.getTaskId()))
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

        ~TaskSendParticlesExchange() override
        {
            notify(this->myId, RECVFINISHED, nullptr);
            if(retryCounter != 0)
            {
                std::cerr << "Performance warning: send/receive buffer for species " << ParBase::FrameType::getName()
                          << " is too small (max: " << maxSize << ", direction: " << exchange << " '"
                          << ExchangeTypeNames{}[exchange] << "'"
                          << ", retries: " << retryCounter << "). To remove this warning consider increasing "
                          << "BYTES_EXCHANGE_{X,Y,Z} in memory.param" << std::endl;
            }
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
        {
            return "TaskSendParticlesExchange";
        }

    private:
        enum state_t
        {
            Constructor,
            Init,
            WaitForBash,
            InitSend,
            WaitForSend,
            WaitForSendEnd,
            Finished
        };


        ParBase& parBase;
        state_t state;
        EventTask tmpEvent;
        EventTask lastSendEvent;
        EventTask initDependency;
        uint32_t exchange;
        size_t maxSize;
        size_t lastSize;
        size_t retryCounter;
    };

} // namespace pmacc

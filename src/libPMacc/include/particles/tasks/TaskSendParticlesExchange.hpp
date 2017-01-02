/**
 * Copyright 2013-2016 Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "eventSystem/EventSystem.hpp"
#include "assert.hpp"

namespace PMacc
{

    template<class ParBase>
    class TaskSendParticlesExchange : public MPITask
    {
    public:

        enum
        {
            Dim = ParBase::Dim,
        };

        TaskSendParticlesExchange(ParBase &parBase, uint32_t exchange) :
        parBase(parBase),
        exchange(exchange),
        state(Constructor),
        maxSize(parBase.getParticlesBuffer().getSendExchangeStack(exchange).getMaxParticlesCount()),
        initDependency(__getTransactionEvent()),
        lastSize(0),lastSendEvent(EventTask()),retryCounter(0){ }

        virtual void init()
        {
            state = Init;
            __startTransaction(initDependency);
            parBase.bashParticles(exchange);
            tmpEvent = __endTransaction();
            state = WaitForBash;
        }

        bool executeIntern()
        {
            switch (state)
            {
                case Init:
                    break;
                case WaitForBash:

                    if (NULL == Environment<>::get().Manager().getITaskIfNotFinished(tmpEvent.getTaskId()) &&
                        NULL == Environment<>::get().Manager().getITaskIfNotFinished(lastSendEvent.getTaskId()))
                    {
                        state = InitSend;
                        //bash is finished
                        __startTransaction();
                        lastSize = parBase.getParticlesBuffer().getSendExchangeStack(exchange).getDeviceParticlesCurrentSize();
                        lastSendEvent = parBase.getParticlesBuffer().asyncSendParticles(__getTransactionEvent(), exchange);
                        initDependency = lastSendEvent;
                        __endTransaction();
                        state = WaitForSend;
                    }

                    break;
                case InitSend:
                    break;
                case WaitForSend:
                    if (NULL == Environment<>::get().Manager().getITaskIfNotFinished(tmpEvent.getTaskId()))
                    {
                        PMACC_ASSERT(lastSize <= maxSize);
                        //check for next bash round
                        if (lastSize == maxSize)
                        {
                            ++retryCounter;
                            init(); //call init and run a full send cycle

                        }
                        else
                            state = WaitForSendEnd;
                    }
                    break;
                case WaitForSendEnd:
                    if (NULL == Environment<>::get().Manager().getITaskIfNotFinished(lastSendEvent.getTaskId()))
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

        virtual ~TaskSendParticlesExchange()
        {
            notify(this->myId, RECVFINISHED, NULL);
            if(retryCounter != 0)
            {
                std::cerr << "Send/receive buffer for species " <<
                    ParBase::FrameType::getName() <<
                    " is to small (max: " << maxSize <<
                    ", direction: " << exchange <<
                    ", retries: " << retryCounter <<
                    ")" << std::endl;
            }
        }

        void event(id_t, EventType, IEventData*) { }

        std::string toString()
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

} //namespace PMacc

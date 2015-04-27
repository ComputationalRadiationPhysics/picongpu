/**
 * Copyright 2013-2014 Rene Widera
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
        lastSize(0),lastSendEvent(EventTask()){ }

        virtual void init()
        {
            state = Init;
            __startTransaction(initDependency);
            parBase.bashParticles(exchange);
            tmpEvent = __endTransaction();
            initDependency=EventTask();
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
                       // std::cout<<"bsend = "<<parBase.getParticlesBuffer().getSendExchangeStack(exchange).getDeviceCurrentSize()<<std::endl;
                        lastSendEvent = parBase.getParticlesBuffer().asyncSendParticles(EventTask(), exchange, tmpEvent);
                        __endTransaction();
                        state = WaitForSend;
                    }

                    break;
                case InitSend:
                    break;
                case WaitForSend:
                    if (NULL == Environment<>::get().Manager().getITaskIfNotFinished(tmpEvent.getTaskId()))
                    {
                        assert(lastSize<=maxSize);
                        //check for next bash round
                        if (lastSize == maxSize)
                        {
                            std::cerr<<"send max size "<<maxSize<<" particles"<<std::endl;
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
    };

} //namespace PMacc

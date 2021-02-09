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
#include "pmacc/traits/NumberOfExchanges.hpp"
#include "pmacc/assert.hpp"

namespace pmacc
{
    template<class ParBase>
    class TaskReceiveParticlesExchange : public MPITask
    {
    public:
        enum
        {
            Dim = ParBase::Dim,
            Exchanges = traits::NumberOfExchanges<Dim>::value
        };

        TaskReceiveParticlesExchange(ParBase& parBase, uint32_t exchange)
            : parBase(parBase)
            , exchange(exchange)
            , state(Constructor)
            , maxSize(parBase.getParticlesBuffer().getReceiveExchangeStack(exchange).getMaxParticlesCount())
            , initDependency(__getTransactionEvent())
            , lastSize(0)
        {
        }

        virtual void init()
        {
            state = Init;
            lastReceiveEvent = parBase.getParticlesBuffer().asyncReceiveParticles(initDependency, exchange);
            initDependency = lastReceiveEvent;
            state = WaitForReceive;
        }

        bool executeIntern()
        {
            switch(state)
            {
            case Init:
                break;
            case WaitForReceive:

                if(nullptr == Environment<>::get().Manager().getITaskIfNotFinished(lastReceiveEvent.getTaskId()))
                {
                    state = InitInsert;
                    // bash is finished
                    __startTransaction();
                    lastSize
                        = parBase.getParticlesBuffer().getReceiveExchangeStack(exchange).getHostParticlesCurrentSize();
                    parBase.insertParticles(exchange);
                    tmpEvent = __endTransaction();
                    initDependency = tmpEvent;
                    state = WaitForInsert;
                }

                break;
            case InitInsert:
                break;
            case WaitForInsert:
                if(nullptr == Environment<>::get().Manager().getITaskIfNotFinished(tmpEvent.getTaskId()))
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

        virtual ~TaskReceiveParticlesExchange()
        {
            notify(this->myId, RECVFINISHED, nullptr);
        }

        void event(id_t, EventType, IEventData*)
        {
        }

        std::string toString()
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

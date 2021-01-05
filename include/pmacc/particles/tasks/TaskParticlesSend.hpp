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

#include "pmacc/Environment.hpp"
#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"

namespace pmacc
{
    template<class T_Particles>
    class TaskParticlesSend : public MPITask
    {
    public:
        typedef T_Particles Particles;
        typedef typename Particles::HandleGuardRegion HandleGuardRegion;
        typedef typename HandleGuardRegion::HandleExchanged HandleExchanged;
        typedef typename HandleGuardRegion::HandleNotExchanged HandleNotExchanged;

        enum
        {
            Dim = Particles::Dim,
            Exchanges = traits::NumberOfExchanges<Dim>::value
        };

        TaskParticlesSend(Particles& parBase) : parBase(parBase), state(Constructor)
        {
        }

        virtual void init()
        {
            state = Init;
            EventTask serialEvent = __getTransactionEvent();
            HandleExchanged handleExchanged;
            HandleNotExchanged handleNotExchanged;

            for(int i = 1; i < Exchanges; ++i)
            {
                /* Start new transaction */
                __startTransaction(serialEvent);

                /* Handle particles */
                if(parBase.getParticlesBuffer().hasSendExchange(i))
                    handleExchanged.handleOutgoing(parBase, i);
                else
                    handleNotExchanged.handleOutgoing(parBase, i);

                /* End transaction */
                tmpEvent += __endTransaction();
            }

            state = WaitForSend;
        }

        bool executeIntern()
        {
            switch(state)
            {
            case Init:
                break;
            case WaitForSend:
                return nullptr == Environment<>::get().Manager().getITaskIfNotFinished(tmpEvent.getTaskId());
            default:
                return false;
            }

            return false;
        }

        virtual ~TaskParticlesSend()
        {
            notify(this->myId, RECVFINISHED, nullptr);
        }

        void event(id_t, EventType, IEventData*)
        {
        }

        std::string toString()
        {
            return "TaskParticlesSend";
        }

    private:
        enum state_t
        {
            Constructor,
            Init,
            WaitForSend

        };


        Particles& parBase;
        state_t state;
        EventTask tmpEvent;
    };

} // namespace pmacc

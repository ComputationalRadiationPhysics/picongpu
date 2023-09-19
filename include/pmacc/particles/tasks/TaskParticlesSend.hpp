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
#include "pmacc/eventSystem/Manager.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"

namespace pmacc
{
    template<class T_Particles>
    class TaskParticlesSend : public MPITask
    {
    public:
        using Particles = T_Particles;
        using HandleGuardRegion = typename Particles::HandleGuardRegion;
        using HandleExchanged = typename HandleGuardRegion::HandleExchanged;
        using HandleNotExchanged = typename HandleGuardRegion::HandleNotExchanged;

        static constexpr uint32_t dim = Particles::dim;

        TaskParticlesSend(Particles& parBase) : parBase(parBase), state(Constructor)
        {
        }

        void init() override
        {
            state = Init;
            EventTask serialEvent = eventSystem::getTransactionEvent();
            HandleExchanged handleExchanged;
            HandleNotExchanged handleNotExchanged;

            static constexpr int32_t numExchanges = traits::NumberOfExchanges<dim>::value;
            for(int i = 1; i < numExchanges; ++i)
            {
                /* Start new transaction */
                eventSystem::startTransaction(serialEvent);

                /* Handle particles */
                if(parBase.getParticlesBuffer().hasSendExchange(i))
                    handleExchanged.handleOutgoing(parBase, i);
                else
                    handleNotExchanged.handleOutgoing(parBase, i);

                /* End transaction */
                tmpEvent += eventSystem::endTransaction();
            }

            state = WaitForSend;
        }

        bool executeIntern() override
        {
            switch(state)
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

        ~TaskParticlesSend() override
        {
            notify(this->myId, RECVFINISHED, nullptr);
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
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

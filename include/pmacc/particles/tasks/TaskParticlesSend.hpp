/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

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
    class TaskParticlesReceive : public MPITask
    {
    public:
        using Particles = T_Particles;
        using HandleGuardRegion = typename Particles::HandleGuardRegion;
        using HandleExchanged = typename HandleGuardRegion::HandleExchanged;
        using HandleNotExchanged = typename HandleGuardRegion::HandleNotExchanged;

        static constexpr uint32_t dim = Particles::dim;

        TaskParticlesReceive(Particles& parBase) : parBase(parBase), state(Constructor)
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
                if(parBase.getParticlesBuffer().hasReceiveExchange(i))
                    handleExchanged.handleIncoming(parBase, i);
                else
                    handleNotExchanged.handleIncoming(parBase, i);

                /* End transaction */
                tmpEvent += eventSystem::endTransaction();
            }

            state = WaitForReceived;
        }

        bool executeIntern() override
        {
            switch(state)
            {
            case Init:
                break;
            case WaitForReceived:
                if(nullptr == Manager::getInstance().getITaskIfNotFinished(tmpEvent.getTaskId()))
                    state = CallFillGaps;
                break;
            case CallFillGaps:
                state = WaitForFillGaps;
                eventSystem::startTransaction();
                parBase.fillBorderGaps();
                tmpEvent = eventSystem::endTransaction();
                state = Finish;
                break;
            case WaitForFillGaps:
                break;
            case Finish:
                return nullptr == Manager::getInstance().getITaskIfNotFinished(tmpEvent.getTaskId());
            default:
                return false;
            }

            return false;
        }

        ~TaskParticlesReceive() override
        {
            notify(this->myId, RECVFINISHED, nullptr);
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
        {
            return "TaskParticlesReceive";
        }

    private:
        enum state_t
        {
            Constructor,
            Init,
            WaitForReceived,
            CallFillGaps,
            WaitForFillGaps,
            Finish
        };

        Particles& parBase;
        state_t state;
        EventTask tmpEvent;
    };

} // namespace pmacc

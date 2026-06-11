/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"

namespace pmacc
{
    template<class TYPE, unsigned DIM>
    class Exchange;

    template<class TYPE, unsigned DIM>
    class TaskSend : public MPITask
    {
    public:
        TaskSend(Exchange<TYPE, DIM>& ex) : exchange(&ex), state(Constructor)
        {
        }

        void init() override
        {
            state = InitDone;
            if(exchange->hasDeviceDoubleBuffer())
            {
                if(Environment<>::get().isMpiDirectEnabled())
                    Environment<>::get().Factory().createTaskCopy(
                        exchange->getDeviceBuffer(),
                        exchange->getDeviceDoubleBuffer(),
                        this);
                else
                {
                    Environment<>::get().Factory().createTaskCopy(
                        exchange->getDeviceBuffer(),
                        exchange->getDeviceDoubleBuffer());

                    Environment<>::get().Factory().createTaskCopy(
                        exchange->getDeviceDoubleBuffer(),
                        exchange->getHostBuffer(),
                        this);
                }
            }
            else
            {
                if(Environment<>::get().isMpiDirectEnabled())
                {
                    /* Wait to be sure that all device work is finished before MPI is triggered.
                     * MPI will not wait for work in our device streams
                     */
                    eventSystem::getTransactionEvent().waitForFinished();
                    state = ReadyForMPISend;
                }
                else
                    Environment<>::get().Factory().createTaskCopy(
                        exchange->getDeviceBuffer(),
                        exchange->getHostBuffer(),
                        this);
            }
        }

        bool executeIntern() override
        {
            switch(state)
            {
            case InitDone:
                break;
            case ReadyForMPISend:
                state = SendDone;
                eventSystem::startTransaction();
                Environment<>::get().Factory().createTaskSendMPI(exchange, this);
                eventSystem::endTransaction();
                break;
            case SendDone:
                break;
            case Finish:
                return true;
            default:
                return false;
            }

            return false;
        }

        ~TaskSend() override
        {
            notify(this->myId, SENDFINISHED, nullptr);
        }

        void event(id_t, EventType type, IEventData*) override
        {
            if(type == COPY)
            {
                state = ReadyForMPISend;
                executeIntern();
            }

            if(type == SENDFINISHED)
            {
                state = Finish;
            }
        }

        std::string toString() override
        {
            std::stringstream ss;
            ss << state;
            return std::string("TaskSend ") + ss.str();
        }

    private:
        enum state_t
        {
            Constructor,
            InitDone,
            ReadyForMPISend,
            SendDone,
            Finish
        };

        Exchange<TYPE, DIM>* exchange;
        state_t state;
    };

} // namespace pmacc

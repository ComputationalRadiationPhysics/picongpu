/* Copyright 2013-2023 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
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

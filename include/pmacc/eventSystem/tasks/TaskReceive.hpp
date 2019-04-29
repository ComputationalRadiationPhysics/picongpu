/* Copyright 2013-2019 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
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

#include "pmacc/eventSystem/tasks/ITask.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"
#include "pmacc/eventSystem/tasks/TaskCopyHostToDevice.hpp"
#include "pmacc/eventSystem/events/EventDataReceive.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/mappings/simulation/EnvironmentController.hpp"
#include "pmacc/memory/buffers/Exchange.hpp"

namespace pmacc
{


    template <class TYPE, unsigned DIM>
    class TaskReceive : public MPITask
    {
    public:

        TaskReceive(Exchange<TYPE, DIM> &ex) :
        exchange(&ex),
        state(Constructor)
        {
        }

        virtual void init()
        {
            state = WaitForReceived;
            Environment<>::get().Factory().createTaskReceiveMPI(exchange, this);
        }

        bool executeIntern()
        {
            switch (state)
            {
                case WaitForReceived:
                    break;
                case RunCopy:
                    state = WaitForFinish;
                   __startTransaction();
                    exchange->getHostBuffer().setCurrentSize(newBufferSize);
                    if (exchange->hasDeviceDoubleBuffer())
                    {

                        Environment<>::get().Factory().createTaskCopyHostToDevice(exchange->getHostBuffer(),
                                                                                     exchange->getDeviceDoubleBuffer());
                        Environment<>::get().Factory().createTaskCopyDeviceToDevice(exchange->getDeviceDoubleBuffer(),
                                                                                       exchange->getDeviceBuffer(),
                                                                                       this);
                    }
                    else
                    {

                        Environment<>::get().Factory().createTaskCopyHostToDevice(exchange->getHostBuffer(),
                                                                                     exchange->getDeviceBuffer(),
                                                                                     this);
                    }
                    __endTransaction();
                    break;
                case WaitForFinish:
                    break;
                case Finish:
                    return true;
                default:
                    return false;
            }

            return false;
        }

        virtual ~TaskReceive()
        {
            notify(this->myId, RECVFINISHED, nullptr);
        }

        void event(id_t, EventType type, IEventData* data)
        {
            switch (type)
            {
                case RECVFINISHED:
                    if (data != nullptr)
                    {
                        EventDataReceive *rdata = static_cast<EventDataReceive*> (data);
                        // std::cout<<" data rec "<<rdata->getReceivedCount()/sizeof(TYPE)<<std::endl;
                        newBufferSize = rdata->getReceivedCount() / sizeof (TYPE);
                        state = RunCopy;
                        executeIntern();
                    }
                    break;
                case COPYHOST2DEVICE:
                case COPYDEVICE2DEVICE:
                    state = Finish;
                    break;
                default:
                    return;
            }
        }

        std::string toString()
        {
            std::stringstream ss;
            ss<<state;
            return std::string("TaskReceive ")+ ss.str();
        }

    private:

        enum state_t
        {
            Constructor,
            WaitForReceived,
            RunCopy,
            WaitForFinish,
            Finish

        };


        Exchange<TYPE, DIM> *exchange;
        state_t state;
        size_t newBufferSize;
    };

} //namespace pmacc


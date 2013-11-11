/**
 * Copyright 2013 Felix Schmitt, René Widera, Wolfgang Hoenig
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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
 

#ifndef _TASKSEND_HPP
#define	_TASKSEND_HPP

#include <cuda_runtime_api.h>

#include "memory/buffers/Exchange.hpp"
#include "eventSystem/EventSystem.hpp"

#include "eventSystem/tasks/ITask.hpp"
#include "eventSystem/tasks/TaskReceive.hpp"
#include "eventSystem/tasks/TaskCopyDeviceToHost.hpp"
#include "mappings/simulation/EnvironmentController.hpp"
#include "eventSystem/tasks/Factory.hpp"

namespace PMacc
{



    template <class TYPE, unsigned DIM>
    class TaskSend : public MPITask
    {
    public:

        TaskSend(Exchange<TYPE, DIM> &ex, EventTask& copyEvent) :
        exchange(&ex),
        copyEvent(copyEvent),
        state(Constructor)
        {
        }

        virtual void init()
        {
            __startTransaction();
            state = InitDone;
            if (exchange->hasDeviceDoubleBuffer())
            {
                Factory::getInstance().createTaskCopyDeviceToDevice(exchange->getDeviceBuffer(),
                                                                               exchange->getDeviceDoubleBuffer()
                                                                               );
                copyEvent = Factory::getInstance().createTaskCopyDeviceToHost(exchange->getDeviceDoubleBuffer(),
                                                                                         exchange->getHostBuffer(),
                                                                                         this);
            }
            else
            {
                copyEvent = Factory::getInstance().createTaskCopyDeviceToHost(exchange->getDeviceBuffer(),
                                                                                         exchange->getHostBuffer(),
                                                                                         this);
            }
            __endTransaction(); //we need no blocking because we get a singnal if transaction is finished
            
        }

        bool executeIntern()
        {
            switch (state)
            {
                case InitDone:
                    break;
                case DeviceToHostFinished:
                    state = SendDone;
                    __startTransaction();
                    Factory::getInstance().createTaskSendMPI(exchange, this);
                    __endTransaction(); //we need no blocking because we get a singnal if transaction is finished
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

        virtual ~TaskSend()
        {
            notify(this->myId, SENDFINISHED, NULL);
        }

        void event(id_t, EventType type, IEventData*)
        {
            if (type == COPYDEVICE2HOST)
            {
                state = DeviceToHostFinished;
                executeIntern();
            }

            if (type == SENDFINISHED)
            {
                state = Finish;
            }

        }

        std::string toString()
        {
            return "TaskSend";
        }

    private:

        enum state_t
        {
            Constructor,
            InitDone,
            DeviceToHostFinished,
            SendDone,
            Finish

        };


        Exchange<TYPE, DIM> *exchange;
        EventTask& copyEvent;
        state_t state;
    };

} //namespace PMacc

#endif	/* _TASKSEND_HPP */


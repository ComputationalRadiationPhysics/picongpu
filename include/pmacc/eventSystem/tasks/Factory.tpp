/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz
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

#include "pmacc/eventSystem/tasks/Factory.hpp"

#include "pmacc/memory/buffers/HostBuffer.hpp"
#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/buffers/Exchange.hpp"

#include "pmacc/eventSystem/tasks/TaskCopyDeviceToHost.hpp"
#include "pmacc/eventSystem/tasks/TaskCopyHostToDevice.hpp"
#include "pmacc/eventSystem/tasks/TaskCopyDeviceToDevice.hpp"
#include "pmacc/eventSystem/tasks/TaskKernel.hpp"
#include "pmacc/eventSystem/tasks/TaskReceive.hpp"
#include "pmacc/eventSystem/tasks/TaskSend.hpp"
#include "pmacc/eventSystem/tasks/TaskSetValue.hpp"
#include "pmacc/eventSystem/tasks/TaskSetCurrentSizeOnDevice.hpp"
#include "pmacc/eventSystem/tasks/TaskSendMPI.hpp"
#include "pmacc/eventSystem/tasks/TaskReceiveMPI.hpp"
#include "pmacc/eventSystem/tasks/TaskGetCurrentSizeFromDevice.hpp"
#include "pmacc/eventSystem/streams/EventStream.hpp"
#include "pmacc/eventSystem/streams/StreamController.hpp"

namespace pmacc
{
    /**
     * creates a TaskCopyHostToDevice
     * @param src HostBuffer to copy data from
     * @param dst DeviceBuffer to copy data to
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template<class TYPE, unsigned DIM>
    inline EventTask Factory::createTaskCopyHostToDevice(
        HostBuffer<TYPE, DIM>& src,
        DeviceBuffer<TYPE, DIM>& dst,
        ITask* registeringTask)
    {
        TaskCopyHostToDevice<TYPE, DIM>* task = new TaskCopyHostToDevice<TYPE, DIM>(src, dst);

        return startTask(*task, registeringTask);
    }

    /**
     * creates a TaskCopyDeviceToHost
     * @param src DeviceBuffer to copy data from
     * @param dst HostBuffer to copy data to
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template<class TYPE, unsigned DIM>
    inline EventTask Factory::createTaskCopyDeviceToHost(
        DeviceBuffer<TYPE, DIM>& src,
        HostBuffer<TYPE, DIM>& dst,
        ITask* registeringTask)
    {
        TaskCopyDeviceToHost<TYPE, DIM>* task = new TaskCopyDeviceToHost<TYPE, DIM>(src, dst);

        return startTask(*task, registeringTask);
    }

    /**
     * creates a TaskCopyDeviceToDevice
     * @param src DeviceBuffer to copy data from
     * @param dst DeviceBuffer to copy data to
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template<class TYPE, unsigned DIM>
    inline EventTask Factory::createTaskCopyDeviceToDevice(
        DeviceBuffer<TYPE, DIM>& src,
        DeviceBuffer<TYPE, DIM>& dst,
        ITask* registeringTask)
    {
        TaskCopyDeviceToDevice<TYPE, DIM>* task = new TaskCopyDeviceToDevice<TYPE, DIM>(src, dst);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a TaskReceive.
     * @param ex Exchange to create new TaskReceive with
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template<class TYPE, unsigned DIM>
    inline EventTask Factory::createTaskReceive(Exchange<TYPE, DIM>& ex, ITask* registeringTask)
    {
        TaskReceive<TYPE, DIM>* task = new TaskReceive<TYPE, DIM>(ex);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a TaskSend.
     * @param ex Exchange to create new TaskSend with
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template<class TYPE, unsigned DIM>
    inline EventTask Factory::createTaskSend(Exchange<TYPE, DIM>& ex, ITask* registeringTask)
    {
        TaskSend<TYPE, DIM>* task = new TaskSend<TYPE, DIM>(ex);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a TaskSendMPI.
     * @param exchange Exchange to create new TaskSendMPI with
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template<class TYPE, unsigned DIM>
    inline EventTask Factory::createTaskSendMPI(Exchange<TYPE, DIM>* ex, ITask* registeringTask)
    {
        TaskSendMPI<TYPE, DIM>* task = new TaskSendMPI<TYPE, DIM>(ex);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a TaskReceiveMPI.
     * @param ex Exchange to create new TaskReceiveMPI with
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template<class TYPE, unsigned DIM>
    inline EventTask Factory::createTaskReceiveMPI(Exchange<TYPE, DIM>* ex, ITask* registeringTask)
    {
        TaskReceiveMPI<TYPE, DIM>* task = new TaskReceiveMPI<TYPE, DIM>(ex);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a new TaskSetValue.
     * @param dst destination DeviceBuffer to set value on
     * @param value value to be set in the DeviceBuffer
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template<class TYPE, unsigned DIM>
    inline EventTask Factory::createTaskSetValue(
        DeviceBuffer<TYPE, DIM>& dst,
        const TYPE& value,
        ITask* registeringTask)
    {
        /* sizeof(TYPE)<256 use fast set method for small data and slow method for big data
         * the rest of 256bytes are reserved for other kernel parameter
         */
        enum
        {
            isSmall = (sizeof(TYPE) <= 128)
        }; // if we use const variable the compiler create warnings

        TaskSetValue<TYPE, DIM, isSmall>* task = new TaskSetValue<TYPE, DIM, isSmall>(dst, value);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a new TaskSetCurrentSizeOnDevice.
     * @param dst destination DeviceBuffer to set current size on
     * @param size size to be set on DeviceBuffer
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template<class TYPE, unsigned DIM>
    inline EventTask Factory::createTaskSetCurrentSizeOnDevice(
        DeviceBuffer<TYPE, DIM>& dst,
        size_t size,
        ITask* registeringTask)
    {
        TaskSetCurrentSizeOnDevice<TYPE, DIM>* task = new TaskSetCurrentSizeOnDevice<TYPE, DIM>(dst, size);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a new TaskGetCurrentSizeFromDevic.
     * @param buffer DeviceBuffer to get current size from
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template<class TYPE, unsigned DIM>
    inline EventTask Factory::createTaskGetCurrentSizeFromDevice(
        DeviceBuffer<TYPE, DIM>& buffer,
        ITask* registeringTask)
    {
        TaskGetCurrentSizeFromDevice<TYPE, DIM>* task = new TaskGetCurrentSizeFromDevice<TYPE, DIM>(buffer);

        return startTask(*task, registeringTask);
    }

    /**
     * Creates a new TaskKernel.
     * @param kernelname name of the kernel which should be called
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     * @return the newly created TaskKernel
     */
    inline TaskKernel* Factory::createTaskKernel(std::string kernelname, ITask* registeringTask)
    {
        TaskKernel* task = new TaskKernel(kernelname);

        if(registeringTask != nullptr)
            task->addObserver(registeringTask);

        return task;
    }


    inline EventTask Factory::startTask(ITask& task, ITask* registeringTask)
    {
        if(registeringTask != nullptr)
        {
            task.addObserver(registeringTask);
        }
        EventTask event(task.getId());

        task.init();
        Environment<>::get().Manager().addTask(&task);
        Environment<>::get().TransactionManager().setTransactionEvent(event);

        return event;
    }


} // namespace pmacc

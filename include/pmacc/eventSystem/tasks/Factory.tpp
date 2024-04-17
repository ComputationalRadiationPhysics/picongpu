/* Copyright 2013-2023 Rene Widera, Benjamin Worpitz
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/eventSystem/Manager.hpp"
#include "pmacc/eventSystem/queues/QueueController.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/eventSystem/tasks/TaskCopy.hpp"
#include "pmacc/eventSystem/tasks/TaskGetCurrentSizeFromDevice.hpp"
#include "pmacc/eventSystem/tasks/TaskKernel.hpp"
#include "pmacc/eventSystem/tasks/TaskReceive.hpp"
#include "pmacc/eventSystem/tasks/TaskReceiveMPI.hpp"
#include "pmacc/eventSystem/tasks/TaskSend.hpp"
#include "pmacc/eventSystem/tasks/TaskSendMPI.hpp"
#include "pmacc/eventSystem/tasks/TaskSetCurrentSizeOnDevice.hpp"
#include "pmacc/eventSystem/tasks/TaskSetValue.hpp"
#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/buffers/Exchange.hpp"
#include "pmacc/memory/buffers/HostBuffer.hpp"

namespace pmacc
{
    /**
     * creates a TaskCopyHostToDevice
     * @param src buffer concept to copy data from
     * @param dst buffer concept to copy data to
     * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
     */
    template<typename T_SrcBuffer, typename T_DestBuffer>
    inline EventTask Factory::createTaskCopy(T_SrcBuffer& src, T_DestBuffer& dst, ITask* registeringTask)
    {
        auto* task = new TaskCopy<T_SrcBuffer, T_DestBuffer>(src, dst);

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
        auto* task = new TaskReceive<TYPE, DIM>(ex);

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
        auto* task = new TaskSend<TYPE, DIM>(ex);

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
        auto* task = new TaskSendMPI<TYPE, DIM>(ex);

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
        auto* task = new TaskReceiveMPI<TYPE, DIM>(ex);

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
        auto* task = new TaskSetValue<TYPE, DIM>(dst, value);

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
        auto* task = new TaskSetCurrentSizeOnDevice<DeviceBuffer<TYPE, DIM>>(dst, size);

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
        auto* task = new TaskGetCurrentSizeFromDevice<DeviceBuffer<TYPE, DIM>>(buffer);

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
        auto* task = new TaskKernel(kernelname);

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
        Manager::getInstance().addTask(&task);
        eventSystem::setTransactionEvent(event);

        return event;
    }


} // namespace pmacc

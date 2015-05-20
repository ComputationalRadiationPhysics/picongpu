/**
 * Copyright 2013 Felix Schmitt, Rene Widera, Wolfgang Hoenig
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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


#ifndef _FACTORY_HPP
#define	_FACTORY_HPP

#include <iostream>
#include "eventSystem/tasks/ITask.hpp"
#include "eventSystem/streams/EventStream.hpp"

namespace PMacc
{
    template <class TYPE, unsigned DIM>
    class HostBuffer;

    template <class TYPE, unsigned DIM>
    class DeviceBuffer;

    template <class TYPE, unsigned DIM>
    class Exchange;

    class TaskKernel;

    /**
     * Singleton Factory-pattern class for creation of several types of EventTasks.
     * Tasks are not actually 'returned' but immediately initialised and
     * added to the Manager's queue. An exception is TaskKernel.
     */
    class Factory
    {
    public:

        /**
         * creates a TaskCopyHostToDevice
         * @param src HostBuffer to copy data from
         * @param dst DeviceBuffer to copy data to
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
         */
        template <class TYPE, unsigned DIM>
        EventTask createTaskCopyHostToDevice(HostBuffer<TYPE, DIM>& src, DeviceBuffer<TYPE, DIM>& dst,
        ITask *registeringTask = NULL);

        /**
         * creates a TaskCopyDeviceToHost
         * @param src DeviceBuffer to copy data from
         * @param dst HostBuffer to copy data to
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
         */
        template <class TYPE, unsigned DIM>
        EventTask createTaskCopyDeviceToHost(DeviceBuffer<TYPE, DIM>& src,
        HostBuffer<TYPE, DIM>& dst,
        ITask *registeringTask = NULL);

        /**
         * creates a TaskCopyDeviceToDevice
         * @param src DeviceBuffer to copy data from
         * @param dst DeviceBuffer to copy data to
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
         */
        template <class TYPE, unsigned DIM>
        EventTask createTaskCopyDeviceToDevice( DeviceBuffer<TYPE, DIM>& src, DeviceBuffer<TYPE, DIM>& dst,
        ITask *registeringTask = NULL);

        /**
         * Creates a TaskReceive.
         * @param ex Exchange to create new TaskReceive with
         * @param task_out returns the newly created task
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
         */
        template <class TYPE, unsigned DIM>
        EventTask createTaskReceive(Exchange<TYPE, DIM> &ex,
        ITask *registeringTask = NULL);

        /**
         * Creates a TaskSend.
         * @param ex Exchange to create new TaskSend with
         * @param task_in TaskReceive to register at new TaskSend
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
         */
        template <class TYPE, unsigned DIM>
        EventTask createTaskSend(Exchange<TYPE, DIM> &ex, EventTask &copyEvent,
        ITask *registeringTask = NULL);

        /**
         * Creates a TaskSendMPI.
         * @param exchange Exchange to create new TaskSendMPI with
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
         */
        template <class TYPE, unsigned DIM>
        EventTask createTaskSendMPI(Exchange<TYPE, DIM> *ex,
        ITask *registeringTask = NULL);

        /**
         * Creates a TaskReceiveMPI.
         * @param ex Exchange to create new TaskReceiveMPI with
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
         */
        template <class TYPE, unsigned DIM>
        EventTask createTaskReceiveMPI(Exchange<TYPE, DIM> *ex,
        ITask *registeringTask = NULL);

        /**
         * Creates a new TaskSetValue.
         * @param dst destination DeviceBuffer to set value on
         * @param value value to be set in the DeviceBuffer
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
         */
        template <class TYPE, unsigned DIM>
        EventTask createTaskSetValue(DeviceBuffer<TYPE, DIM>& dst, const TYPE& value,
        ITask *registeringTask = NULL);

        /**
         * Creates a new TaskSetCurrentSizeOnDevice.
         * @param dst destination DeviceBuffer to set current size on
         * @param size size to be set on DeviceBuffer
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
         */
        template <class TYPE, unsigned DIM>
        EventTask createTaskSetCurrentSizeOnDevice(DeviceBuffer<TYPE, DIM>& dst, size_t size,
        ITask *registeringTask = NULL);

        /**
         * Creates a new TaskGetCurrentSizeFromDevic.
         * @param buffer DeviceBuffer to get current size from
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
         */
        template <class TYPE, unsigned DIM>
        EventTask createTaskGetCurrentSizeFromDevice(DeviceBuffer<TYPE, DIM>& buffer,
        ITask *registeringTask = NULL);

        /**
         * Creates a new TaskKernel.
         * @param kernelname name of the kernel which should be called
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
         * @return the newly created TaskKernel
         */
        TaskKernel* createTaskKernel(std::string kernelname, ITask *registeringTask = NULL);

        /**
         * Starts a task by initialising it and adding it to the Manager's queue.
         *
         * @param task the ITask to start
         * @param registeringTask optional task which can be registered as an observer for task
         */
        EventTask startTask(ITask& task, ITask *registeringTask);

    private:

        friend class Environment<DIM1>;
        friend class Environment<DIM2>;
        friend class Environment<DIM3>;

        Factory() {};

        Factory(const Factory&) { };

        static Factory& getInstance()
        {
            static Factory instance;
            return instance;
        }

    };

} //namespace PMacc


#endif	/* _FACTORY_HPP */


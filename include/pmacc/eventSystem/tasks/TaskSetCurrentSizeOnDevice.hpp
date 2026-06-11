/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/events/kernelEvents.hpp"
#include "pmacc/eventSystem/tasks/DeviceTask.hpp"

namespace pmacc
{
    struct KernelSetValueOnDeviceMemory
    {
        template<typename T_Worker>
        DINLINE void operator()(T_Worker const&, size_t* pointer, size_t const size) const
        {
            *pointer = size;
        }
    };

    template<typename T_DeviceBuffer>
    class TaskSetCurrentSizeOnDevice : public DeviceTask
    {
    public:
        TaskSetCurrentSizeOnDevice(T_DeviceBuffer& dst, size_t size) : DeviceTask(), destination(&dst), size(size)
        {
        }

        ~TaskSetCurrentSizeOnDevice() override
        {
            notify(this->myId, SETVALUE, nullptr);
        }

        void init() override
        {
            setSize();
        }

        bool executeIntern() override
        {
            return isFinished();
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
        {
            return "TaskSetCurrentSizeOnDevice";
        }

    private:
        void setSize()
        {
            auto sizeBuff = destination->sizeOnDeviceBuffer();

            auto alpakaAllOne = DataSpace<DIM1>(1).toAlpakaKernelVec();
            auto oneThread
                = alpaka::WorkDivMembers<AlpakaDim<DIM1>, IdxType>{alpakaAllOne, alpakaAllOne, alpakaAllOne};
            auto setValueKernel = alpaka::createTaskKernel<Acc<DIM1>>(
                oneThread,
                KernelSetValueOnDeviceMemory{},
                alpaka::getPtrNative(sizeBuff),
                size);
            auto queue = this->getAlpakaQueue();
            alpaka::enqueue(queue, setValueKernel);

            activate();
        }

        T_DeviceBuffer* destination;
        size_t const size;
    };

} // namespace pmacc

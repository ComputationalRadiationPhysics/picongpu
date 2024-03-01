/* Copyright 2015-2023 Rene Widera, Alexander Grund
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

#if(ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED)

#    include "pmacc/alpakaHelper/Device.hpp"
#    include "pmacc/eventSystem/eventSystem.hpp"
#    include "pmacc/math/Vector.hpp"
#    include "pmacc/particles/memory/buffers/MallocMCBuffer.hpp"
#    include "pmacc/types.hpp"

#    include <memory>


namespace pmacc
{
    template<typename T_DeviceHeap>
    MallocMCBuffer<T_DeviceHeap>::MallocMCBuffer(const std::shared_ptr<DeviceHeap>& deviceHeap)
        : /* currently mallocMC has only one heap */
        deviceHeapInfo(deviceHeap->getHeapLocations()[0])
        , hostBufferOffset(0)
    {
    }

    template<typename T_DeviceHeap>
    MallocMCBuffer<T_DeviceHeap>::~MallocMCBuffer()
    {
    }

    template<typename T_DeviceHeap>
    void MallocMCBuffer<T_DeviceHeap>::synchronize()
    {
        auto alpakaBufferSize = pmacc::math::Vector<pmacc::MemIdxType, 1>(deviceHeapInfo.size).toAlpakaMemVec();

        if(!hostBuffer)
        {
            hostBuffer = alpaka::allocMappedBufIfSupported<uint8_t, MemIdxType>(
                manager::Device<HostDevice>::get().current(),
                manager::Device<ComputeDevice>::get().getPlatform(),
                alpakaBufferSize);

            hostBufferOffset = static_cast<int64_t>(
                reinterpret_cast<uint8_t*>(deviceHeapInfo.p) - alpaka::getPtrNative(*hostBuffer));
        }
        /* add event system hints */
        eventSystem::startOperation(ITask::TASK_DEVICE);
        eventSystem::startOperation(ITask::TASK_HOST);

        auto devView = ::alpaka::ViewPlainPtr<ComputeDevice, uint8_t, AlpakaDim<DIM1>, pmacc::MemIdxType>(
            (uint8_t*) deviceHeapInfo.p,
            manager::Device<ComputeDevice>::get().current(),
            alpakaBufferSize);
        auto alpakaStream = pmacc::eventSystem::getEventStream(ITask::TASK_DEVICE)->getCudaStream();
        alpaka::memcpy(alpakaStream, *hostBuffer, devView, alpakaBufferSize);
        alpaka::wait(alpakaStream);
    }

} // namespace pmacc

#endif

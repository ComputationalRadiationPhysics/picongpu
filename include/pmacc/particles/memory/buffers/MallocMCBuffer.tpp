/* Copyright 2015-2021 Rene Widera, Alexander Grund
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

#if(PMACC_CUDA_ENABLED == 1 || ALPAKA_ACC_GPU_HIP_ENABLED == 1)

#    include "pmacc/particles/memory/buffers/MallocMCBuffer.hpp"
#    include "pmacc/types.hpp"
#    include "pmacc/eventSystem/EventSystem.hpp"

#    include <memory>


namespace pmacc
{
    template<typename T_DeviceHeap>
    MallocMCBuffer<T_DeviceHeap>::MallocMCBuffer(const std::shared_ptr<DeviceHeap>& deviceHeap)
        : hostPtr(nullptr)
        ,
        /* currently mallocMC has only one heap */
        deviceHeapInfo(deviceHeap->getHeapLocations()[0])
        , hostBufferOffset(0)
    {
    }

    template<typename T_DeviceHeap>
    MallocMCBuffer<T_DeviceHeap>::~MallocMCBuffer()
    {
        if(hostPtr != nullptr)
        {
#    if(PMACC_CUDA_ENABLED == 1)
            cudaHostUnregister(hostPtr);
            __deleteArray(hostPtr);
#    else
            CUDA_CHECK_NO_EXCEPT((cuplaError_t) hipFree(hostPtr));
#    endif
        }
    }

    template<typename T_DeviceHeap>
    void MallocMCBuffer<T_DeviceHeap>::synchronize()
    {
        /** \todo: we had no abstraction to create a host buffer and a pseudo
         *         device buffer (out of the mallocMC ptr) and copy both with our event
         *         system.
         *         WORKAROUND: use native CUDA/HIP calls :-(
         */
        if(hostPtr == nullptr)
        {
#    if(PMACC_CUDA_ENABLED == 1)
            /* use `new` and than `cudaHostRegister` is faster than `cudaMallocHost`
             * but with the some result (create page-locked memory)
             */
            hostPtr = new char[deviceHeapInfo.size];
            CUDA_CHECK((cuplaError_t) cudaHostRegister(hostPtr, deviceHeapInfo.size, cudaHostRegisterDefault));
#    else
            // we do not use hipHostRegister because this would require a strict alignment
            // https://github.com/alpaka-group/alpaka/pull/896
            CUDA_CHECK((cuplaError_t) hipHostMalloc((void**) &hostPtr, deviceHeapInfo.size, hipHostMallocDefault));
#    endif

            this->hostBufferOffset = static_cast<int64_t>(reinterpret_cast<char*>(deviceHeapInfo.p) - hostPtr);
        }
        /* add event system hints */
        __startOperation(ITask::TASK_DEVICE);
        __startOperation(ITask::TASK_HOST);
        CUDA_CHECK(cuplaMemcpy(hostPtr, deviceHeapInfo.p, deviceHeapInfo.size, cuplaMemcpyDeviceToHost));
    }

} // namespace pmacc

#endif

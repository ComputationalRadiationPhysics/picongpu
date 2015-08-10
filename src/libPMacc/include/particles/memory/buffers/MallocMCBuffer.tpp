/**
 * Copyright 2015 Rene Widera, Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
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

#pragma once

#include "particles/memory/buffers/MallocMCBuffer.hpp"
#include "types.h"
#include "Environment.hpp"
#include "eventSystem/EventSystem.hpp"

namespace PMacc
{

MallocMCBuffer::MallocMCBuffer( ) : hostPtr( NULL ),hostBufferOffset(0)
{
    /* currently mallocMC has only one heap */
    this->deviceHeapInfo=mallocMC::getHeapLocations()[0];
    Environment<>::get().DataConnector().registerData( *this);
}

MallocMCBuffer::~MallocMCBuffer( )
{
    if ( hostPtr != NULL )
        cudaHostUnregister(hostPtr);

    __deleteArray(hostPtr);

}

void MallocMCBuffer::synchronize( )
{
    /** \todo: we had no abstraction to create a host buffer and a pseudo
     *         device buffer (out of the mallocMC ptr) and copy both with our event
     *         system.
     *         WORKAROUND: use native cuda calls :-(
     */
    if ( hostPtr == NULL )
    {
        /* use `new` and than `cudaHostRegister` is faster than `cudaMallocHost`
         * but with the some result (create page-locked memory)
         */
        hostPtr = new char[deviceHeapInfo.size];
        CUDA_CHECK(cudaHostRegister(hostPtr, deviceHeapInfo.size, cudaHostRegisterDefault));


        this->hostBufferOffset = static_cast<int64_t>(reinterpret_cast<char*>(deviceHeapInfo.p) - hostPtr);
    }
    /* add event system hints */
    __startOperation(ITask::TASK_CUDA);
    __startOperation(ITask::TASK_HOST);
    CUDA_CHECK(cudaMemcpy(hostPtr, deviceHeapInfo.p, deviceHeapInfo.size, cudaMemcpyDeviceToHost));

}

} //namespace PMacc

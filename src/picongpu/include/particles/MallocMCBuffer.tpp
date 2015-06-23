/**
 * Copyright 2015 Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "particles/MallocMCBuffer.hpp"

namespace picongpu
{
using namespace PMacc;

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
        CUDA_CHECK(cudaHostRegister(hostPtr,deviceHeapInfo.size,cudaHostRegisterDefault));


        this->hostBufferOffset=int64_t(((char*)deviceHeapInfo.p) - hostPtr);
    }
    /* add event system hints */
    __startOperation(ITask::TASK_CUDA);
    __startOperation(ITask::TASK_HOST);
    CUDA_CHECK(cudaMemcpy(hostPtr,deviceHeapInfo.p,deviceHeapInfo.size,cudaMemcpyDeviceToHost));

}

} //namespace picongpu

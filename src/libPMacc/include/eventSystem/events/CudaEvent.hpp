/**
 * Copyright 2014 Rene Widera
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

#include "types.h"
#include <cuda_runtime.h>

namespace PMacc
{

/**
 * Wrapper for cuda events
 */
class CudaEvent
{
private:

    cudaEvent_t event;
    cudaStream_t stream;
    /* state if event is recorded */
    bool isRecorded;
    bool isValid;


public:

    /**
     * Constructor
     *
     * no data is allocated @see create()
     */
    CudaEvent() : isRecorded(false), isValid(false)
    {

    }

    /**
     * Destructor
     *
     * no data is freed @see destroy()
     */
    virtual ~CudaEvent()
    {

    }

    /**
     *  create valid object
     *
     * - internal memory is allocated
     * - event must be destroyed with @see destroy
     */
    static CudaEvent create()
    {
        CudaEvent ev;
        ev.isValid = true;
        CUDA_CHECK(cudaEventCreateWithFlags(&(ev.event), cudaEventDisableTiming));
        return ev;
    }

    /**
     * free allocated memory
     */
    static void destroy(const CudaEvent& ev)
    {
        CUDA_CHECK(cudaEventSynchronize(ev.event));
        CUDA_CHECK(cudaEventDestroy(ev.event));
    }

    /**
     * get native cuda event
     *
     * @return native cuda event
     */
    cudaEvent_t operator*() const
    {
        assert(isValid);
        return event;
    }

    /**
     * check whether the event is finished
     *
     * @return true if event is finished else false
     */
    bool isFinished() const
    {
        assert(isValid);
        return cudaEventQuery(event) == cudaSuccess;
    }


    /**
     * get stream in which this event is recorded
     *
     * @return native cuda stream
     */
    cudaStream_t getStream() const
    {
        assert(isRecorded);
        return stream;
    }

    /**
     * record event in a device stream
     *
     * @param stream native cuda stream
     */
    void recordEvent(cudaStream_t stream)
    {
        /* disallow double recording */
        assert(isRecorded==false);
        isRecorded = true;
        this->stream = stream;
        CUDA_CHECK(cudaEventRecord(event, stream));
    }

};
}

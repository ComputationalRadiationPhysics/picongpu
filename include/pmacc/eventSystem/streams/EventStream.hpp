/* Copyright 2013-2019 Felix Schmitt, Rene Widera, Benjamin Worpitz
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

#include "pmacc/eventSystem/events/CudaEventHandle.hpp"
#include "pmacc/types.hpp"



namespace pmacc
{

/**
 * Wrapper for a single cuda stream.
 * Allows recording cuda events on the stream.
 */
class EventStream
{
public:

    /**
     * Constructor.
     * Creates the cudaStream_t object.
     */
    EventStream() : stream(nullptr)
    {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    /**
     * Destructor.
     * Waits for the stream to finish and destroys it.
     */
    virtual ~EventStream()
    {
        // wait for all kernels in stream to finish
        CUDA_CHECK_NO_EXCEPT(cudaStreamSynchronize(stream));
        CUDA_CHECK_NO_EXCEPT(cudaStreamDestroy(stream));
    }

    /**
     * Returns the cudaStream_t object associated with this EventStream.
     * @return the internal cuda stream object
     */
    cudaStream_t getCudaStream() const
    {
        return stream;
    }

    void waitOn(const CudaEventHandle& ev)
    {
        if (this->stream != ev.getStream())
        {
            CUDA_CHECK(cudaStreamWaitEvent(this->getCudaStream(), *ev, 0));
        }
    }

private:
    cudaStream_t stream;
};

}

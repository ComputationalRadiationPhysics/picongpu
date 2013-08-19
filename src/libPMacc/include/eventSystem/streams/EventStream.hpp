/**
 * Copyright 2013 Felix Schmitt, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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
 

#ifndef EVENTSTREAM_HPP
#define	EVENTSTREAM_HPP

#include <cuda_runtime.h>

namespace PMacc
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
        EventStream(): stream(NULL)
        {
            CUDA_CHECK(cudaStreamCreate(&stream));
        }

        /**
         * Destructor.
         * Waits for the stream to finish and destroys it.
         */
        virtual ~EventStream()
        {
            //wait for all kernels in stream to finish
            CUDA_CHECK(cudaStreamSynchronize(stream)); 
            CUDA_CHECK(cudaStreamDestroy(stream));
        }

        /**
         * Records a cudaEvent_t at the end of this stream.
         * @param event the cuda event to record
         */
        void recordEvent(cudaEvent_t event)
        {
            assert(stream!=NULL);
            CUDA_CHECK(cudaEventRecord(event, stream));
        }

        /**
         * Returns the cudaStream_t object associated with this EventStream.
         * @return the internal cuda stream object
         */
        cudaStream_t getCudaStream() const
        {
            return stream;
        }

    private:
        cudaStream_t stream;
    };

}

#endif	/* EVENTSTREAM_HPP */


/* Copyright 2013-2022 Felix Schmitt, Rene Widera, Benjamin Worpitz
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

#include "pmacc/eventSystem/streams/EventStream.hpp"

namespace pmacc
{
    EventStream::EventStream()
    {
        CUDA_CHECK(cuplaStreamCreate(&stream));
    }

    EventStream::~EventStream()
    {
        // wait for all kernels in stream to finish
        CUDA_CHECK_NO_EXCEPT(cuplaStreamSynchronize(stream));
        CUDA_CHECK_NO_EXCEPT(cuplaStreamDestroy(stream));
    }

    cuplaStream_t EventStream::getCudaStream() const
    {
        return stream;
    }

    void EventStream::waitOn(const CudaEventHandle& ev)
    {
        if(this->stream != ev.getStream())
        {
            CUDA_CHECK(cuplaStreamWaitEvent(this->getCudaStream(), *ev, 0));
        }
    }
} // namespace pmacc

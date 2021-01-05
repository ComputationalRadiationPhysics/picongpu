/* Copyright 2016-2021 Rene Widera
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

#include "pmacc/eventSystem/events/CudaEvent.def"
#include "pmacc/eventSystem/events/CudaEventHandle.hpp"
#include "pmacc/types.hpp"


namespace pmacc
{
    CudaEvent::CudaEvent() : isRecorded(false), finished(true), refCounter(0u)
    {
        log(ggLog::CUDA_RT() + ggLog::EVENT(), "create event");
        CUDA_CHECK(cuplaEventCreateWithFlags(&event, cuplaEventDisableTiming));
    }


    CudaEvent::~CudaEvent()
    {
        PMACC_ASSERT(refCounter == 0u);
        log(ggLog::CUDA_RT() + ggLog::EVENT(), "sync and delete event");
        // free cupla event
        CUDA_CHECK_NO_EXCEPT(cuplaEventSynchronize(event));
        CUDA_CHECK_NO_EXCEPT(cuplaEventDestroy(event));
    }

    void CudaEvent::registerHandle()
    {
        ++refCounter;
    }

    void CudaEvent::releaseHandle()
    {
        assert(refCounter != 0u);
        // get old value and decrement
        uint32_t oldCounter = refCounter--;
        if(oldCounter == 1u)
        {
            // reset event meta data
            isRecorded = false;
            finished = true;

            Environment<>::get().EventPool().push(this);
        }
    }


    bool CudaEvent::isFinished()
    {
        // avoid cupla driver calls if event is already finished
        if(finished)
            return true;
        assert(isRecorded);

        cuplaError_t rc = cuplaEventQuery(event);

        if(rc == cuplaSuccess)
        {
            finished = true;
            return true;
        }
        else if(rc == cuplaErrorNotReady)
            return false;
        else
            PMACC_PRINT_CUPLA_ERROR_AND_THROW(rc, "Event query failed");
    }


    void CudaEvent::recordEvent(cuplaStream_t stream)
    {
        /* disallow double recording */
        assert(isRecorded == false);
        isRecorded = true;
        finished = false;
        this->stream = stream;
        CUDA_CHECK(cuplaEventRecord(event, stream));
    }

} // namespace pmacc

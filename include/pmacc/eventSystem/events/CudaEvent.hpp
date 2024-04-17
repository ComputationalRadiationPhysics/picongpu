/* Copyright 2016-2023 Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/types.hpp"


namespace pmacc
{
    /** Wrapper for ComputeEvent
     *
     * This class follows the RAII rules
     */
    class CudaEvent
    {
    private:
        /** native alpaka event */
        ComputeEvent event;
        /** native alpaka queue where the event is recorded
         *
         *  only valid if isRecorded is true
         */
        std::optional<ComputeQueue> stream;
        /** state if a recorded event is finished
         *
         * avoids that alpaka calls backend API methods after `isFinished()` returns the first time true
         */
        bool finished{true};

        /** number of CudaEventHandle's to the instance */
        uint32_t refCounter{0u};


    public:
        /** Constructor
         *
         * if called before the alpaka device is initialized the behavior is undefined
         */
        CudaEvent();

        /** Destructor */
        ~CudaEvent();

        /** register a existing handle to a event instance */
        void registerHandle();

        /** free a registered handle */
        void releaseHandle();

        /** get native ComputeEvent object
         *
         * @return native alpaka event
         */
        ComputeEvent operator*() const
        {
            return event;
        }

        /** get stream in which this event is recorded
         *
         * @return native alpaka queue
         */
        ComputeQueue getStream() const
        {
            assert(this->stream.has_value());
            return *stream;
        }

        /** check whether the event is finished
         *
         * @return true if event is finished else false
         */
        bool isFinished();

        /** record event in a device stream
         *
         * @param stream native alpaka queue
         */
        void recordEvent(ComputeQueue const& stream);
    };
} // namespace pmacc

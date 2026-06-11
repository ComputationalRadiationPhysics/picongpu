/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /** Wrapper for ComputeDeviceEvent
     *
     * This class follows the RAII rules
     */
    class ComputeEvent
    {
    private:
        /** native alpaka event */
        ComputeDeviceEvent event;
        /** native alpaka queue where the event is recorded
         *
         *  only valid if isRecorded is true
         */
        std::optional<ComputeDeviceQueue> stream;
        /** state if a recorded event is finished
         *
         * avoids that alpaka calls backend API methods after `isFinished()` returns the first time true
         */
        bool finished{true};

        /** number of ComputeEventHandle's to the instance */
        uint32_t refCounter{0u};


    public:
        /** Constructor
         *
         * if called before the alpaka device is initialized the behavior is undefined
         */
        ComputeEvent();

        /** Destructor */
        ~ComputeEvent();

        /** register a existing handle to a event instance */
        void registerHandle();

        /** free a registered handle */
        void releaseHandle();

        /** get native ComputeDeviceEvent object
         *
         * @return native alpaka event
         */
        ComputeDeviceEvent operator*() const
        {
            return event;
        }

        /** get stream in which this event is recorded
         *
         * @return native alpaka queue
         */
        ComputeDeviceQueue getStream() const
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
        void recordEvent(ComputeDeviceQueue const& stream);
    };
} // namespace pmacc

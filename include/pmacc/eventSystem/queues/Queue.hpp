/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/eventSystem/events/ComputeEventHandle.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /** Wrapper for a single alpaka queue.
     *
     * Allows recording alpaka events on the queue.
     */
    class Queue
    {
    public:
        Queue();

        /** Destructor.
         *
         * Waits for the queue to finish and destroys it.
         */
        virtual ~Queue();

        /** Returns the alpaka queue object associated with this Queue.
         *
         * @return the internal alpaka queue object
         */
        ComputeDeviceQueue getAlpakaQueue() const;

        void waitOn(ComputeEventHandle const& ev);

    private:
        ComputeDeviceQueue queue;
    };

} // namespace pmacc

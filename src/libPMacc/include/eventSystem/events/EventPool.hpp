/**
 * Copyright 2013 Felix Schmitt, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
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


#ifndef EVENTPOOL_HPP
#define	EVENTPOOL_HPP

#include "types.h"
#include <vector>
#include <cuda_runtime.h>

namespace PMacc
{

    /**
     * Manages a pool of cudaEvent_t objects and gives access to them.
     */
    class EventPool
    {
    public:

        /**
         * Constructor.
         * adds a single cuda event to the pool
         */
        EventPool()
        {
            addEvents(1);
            currentEventIndex = 0;
        }

        /**
         * Destructor.
         * destroys all cuda events in the pool
         */
        virtual ~EventPool()
        {
            for (size_t i = 0; i < events.size(); i++)
            {
                log(ggLog::CUDA_RT()+ggLog::MEMORY(),"Sync and Delete Event: %1%") % i;
                CUDA_CHECK(cudaEventSynchronize(events[i]));
                CUDA_CHECK(cudaEventDestroy(events[i]));
            }
        }

        /**
         * Returns the next cuda event in the event pool.
         * @return the next cuda event
         */
        cudaEvent_t getNextEvent()
        {
            cudaEvent_t result = events[currentEventIndex];
            currentEventIndex = (currentEventIndex + 1) % events.size();

            return result;
        }

        /**
         * Adds count new cuda events to the pool.
         * @param count number of cuda events to add
         */
        void addEvents(size_t count)
        {
            for (size_t i = 0; i < count; i++)
            {
                cudaEvent_t event;
                CUDA_CHECK(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
                events.push_back(event);
            }
        }

        /**
         * Returns the number of cuda events in the pool.
         * @return number of cuda events
         */
        size_t getEventsCount()
        {
            return events.size();
        }

    private:
        std::vector<cudaEvent_t> events;
        size_t currentEventIndex;
    };
}

#endif	/* EVENTPOOL_HPP */


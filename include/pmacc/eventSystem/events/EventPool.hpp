/* Copyright 2013-2023 Felix Schmitt, Rene Widera, Benjamin Worpitz
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

#include "pmacc/Environment.def"
#include "pmacc/debug/VerboseLog.hpp"
#include "pmacc/eventSystem/events/CudaEvent.hpp"
#include "pmacc/eventSystem/events/CudaEventHandle.hpp"
#include "pmacc/types.hpp"

#include <list>
#include <stdexcept>
#include <vector>

namespace pmacc
{
    /** Manages a pool of EventType objects and gives access to them. */
    class EventPool
    {
    public:
        /** Returns a free cupla event
         *
         * @return free cupla event
         */
        CudaEventHandle pop()
        {
            if(freeEvents.size() != 0)
            {
                CudaEventHandle result = freeEvents.front();
                freeEvents.pop_front();
                return result;
            }
            createEvents();
            return pop();
        }


        /** add CudaEvent to the pool
         *
         * the pool takes the ownership of the pointer
         *
         * @param ev pointer to CudaEvent
         */
        void push(CudaEvent* const ev)
        {
            /* Guard that no event is added during the pool is closed (shutdown phase).
             * This method is also called during the evaluation of the destructor.
             */
            if(!isClosed)
                freeEvents.push_back(CudaEventHandle(ev));
        }

        /** create and add cupla events to the pool
         *
         * @param count number of cupla events to add
         */
        void createEvents(size_t count = 1u)
        {
            for(size_t i = 0u; i < count; i++)
            {
                auto* nativeEvent = new CudaEvent();
                events.push_back(nativeEvent);
                push(nativeEvent);
            }
        }

        /** Returns the number of cupla events in the pool.
         *
         * @return number of cupla events
         */
        size_t getEventsCount()
        {
            return events.size();
        }

    private:
        friend struct detail::Environment;

        static EventPool& getInstance()
        {
            static EventPool instance;
            return instance;
        }

        /** Constructor */
        EventPool() = default;

        /** Destructor
         *
         * destroys all cupla events in the pool
         */
        ~EventPool()
        {
            log(ggLog::CUDA_RT() + ggLog::EVENT(), "shutdown EventPool with %1% events") % getEventsCount();
            isClosed = true;
            freeEvents.clear();
            for(std::vector<CudaEvent*>::const_iterator iter = events.begin(); iter != events.end(); ++iter)
            {
                delete *iter;
            }
            events.clear();
        }

        //! hold all CudaEvents
        std::vector<CudaEvent*> events;

        //! hold currently free CudaEventHandle's
        std::list<CudaEventHandle> freeEvents;

        /**! state if the pool is closed
         *
         * if true no events can be added to the pool
         */
        bool isClosed{false};
    };
} // namespace pmacc

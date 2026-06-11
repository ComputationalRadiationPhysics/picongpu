/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/Environment.def"
#include "pmacc/debug/PMaccVerbose.hpp"
#include "pmacc/eventSystem/events/ComputeEvent.hpp"
#include "pmacc/eventSystem/events/ComputeEventHandle.hpp"
#include "pmacc/types.hpp"

#include <list>
#include <vector>

namespace pmacc
{
    /** Manages a pool of EventType objects and gives access to them. */
    class EventPool
    {
    public:
        /** Returns a free alpaka event
         *
         * @return free alpaka event
         */
        ComputeEventHandle pop()
        {
            if(freeEvents.size() != 0)
            {
                ComputeEventHandle result = freeEvents.front();
                freeEvents.pop_front();
                return result;
            }
            createEvents();
            return pop();
        }

        /** add ComputeEvent to the pool
         *
         * the pool takes the ownership of the pointer
         *
         * @param ev pointer to ComputeEvent
         */
        void push(ComputeEvent* const ev)
        {
            /* Guard that no event is added during the pool is closed (shutdown phase).
             * This method is also called during the evaluation of the destructor.
             */
            if(!isClosed)
                freeEvents.push_back(ComputeEventHandle(ev));
        }

        /** create and add an alpaka events to the pool
         *
         * @param count number of alpaka events to add
         */
        void createEvents(size_t count = 1u)
        {
            for(size_t i = 0u; i < count; i++)
            {
                auto* nativeEvent = new ComputeEvent();
                events.push_back(nativeEvent);
                push(nativeEvent);
            }
        }

        /** Returns the number of alpaka events in the pool.
         *
         * @return number of alpaka events
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
         * destroys all alpaka events in the pool
         */
        ~EventPool()
        {
            log(ggLog::CUDA_RT() + ggLog::EVENT(), "shutdown EventPool with %1% events") % getEventsCount();
            isClosed = true;
            freeEvents.clear();
            for(std::vector<ComputeEvent*>::const_iterator iter = events.begin(); iter != events.end(); ++iter)
            {
                delete *iter;
            }
            events.clear();
        }

        //! hold all ComputeEvents
        std::vector<ComputeEvent*> events;

        //! hold currently free ComputeEventHandle's
        std::list<ComputeEventHandle> freeEvents;

        /**! state if the pool is closed
         *
         * if true no events can be added to the pool
         */
        bool isClosed{false};
    };
} // namespace pmacc

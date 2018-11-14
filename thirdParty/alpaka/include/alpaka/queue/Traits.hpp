/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/wait/Traits.hpp>

#include <alpaka/core/Common.hpp>

#include <type_traits>
#include <utility>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The queue specifics.
    namespace queue
    {
        //-----------------------------------------------------------------------------
        //! The queue traits.
        namespace traits
        {
            //#############################################################################
            //! The queue enqueue trait.
            template<
                typename TQueue,
                typename TTask,
                typename TSfinae = void>
            struct Enqueue;

            //#############################################################################
            //! The queue empty trait.
            template<
                typename TQueue,
                typename TSfinae = void>
            struct Empty;
        }

        //-----------------------------------------------------------------------------
        //! Queues the given task in the given queue.
        //!
        //! Special Handling for events:
        //!   If the event has previously been queued, then this call will overwrite any existing state of the event.
        //!   Any subsequent calls which examine the status of event will only examine the completion of this most recent call to enqueue.
        template<
            typename TQueue,
            typename TTask>
        ALPAKA_FN_HOST auto enqueue(
            TQueue & queue,
            TTask && task)
        -> void
        {
            traits::Enqueue<
                TQueue,
                typename std::decay<TTask>::type>
            ::enqueue(
                queue,
                std::forward<TTask>(task));
        }

        //-----------------------------------------------------------------------------
        //! Tests if the queue is empty (all ops in the given queue have been completed).
        template<
            typename TQueue>
        ALPAKA_FN_HOST auto empty(
            TQueue const & queue)
        -> bool
        {
            return
                traits::Empty<
                    TQueue>
                ::empty(
                    queue);
        }
    }
}

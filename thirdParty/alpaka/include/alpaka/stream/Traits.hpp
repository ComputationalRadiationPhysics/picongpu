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

#include <alpaka/wait/Traits.hpp>   // CurrentThreadWaitFor, WaiterWaitFor

#include <alpaka/core/Common.hpp>   // ALPAKA_FN_*

#include <type_traits>              // std::decay
#include <utility>                  // std::forward

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The stream specifics.
    //-----------------------------------------------------------------------------
    namespace stream
    {
        //-----------------------------------------------------------------------------
        //! The stream traits.
        //-----------------------------------------------------------------------------
        namespace traits
        {
            //#############################################################################
            //! The stream enqueue trait.
            //#############################################################################
            template<
                typename TStream,
                typename TTask,
                typename TSfinae = void>
            struct Enqueue;

            //#############################################################################
            //! The stream empty trait.
            //#############################################################################
            template<
                typename TStream,
                typename TSfinae = void>
            struct Empty;
        }

        //-----------------------------------------------------------------------------
        //! Queues the given task in the given stream.
        //!
        //! If it has previously been queued, then this call will overwrite any existing state of the event.
        //! Any subsequent calls which examine the status of event will only examine the completion of this most recent call to enqueue.
        //-----------------------------------------------------------------------------
        template<
            typename TStream,
            typename TTask>
        ALPAKA_FN_HOST auto enqueue(
            TStream & stream,
            TTask && task)
        -> void
        {
            traits::Enqueue<
                TStream,
                typename std::decay<TTask>::type>
            ::enqueue(
                stream,
                std::forward<TTask>(task));
        }

        //-----------------------------------------------------------------------------
        //! Tests if the stream is empty (all ops in the given stream have been completed).
        //-----------------------------------------------------------------------------
        template<
            typename TStream>
        ALPAKA_FN_HOST auto empty(
            TStream const & stream)
        -> bool
        {
            return
                traits::Empty<
                    TStream>
                ::empty(
                    stream);
        }
    }
}

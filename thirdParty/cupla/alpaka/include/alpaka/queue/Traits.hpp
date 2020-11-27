/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/wait/Traits.hpp>

#include <type_traits>
#include <utility>

namespace alpaka
{
    struct ConceptQueue;

    //-----------------------------------------------------------------------------
    //! The queue traits.
    namespace traits
    {
        //#############################################################################
        //! The queue enqueue trait.
        template<typename TQueue, typename TTask, typename TSfinae = void>
        struct Enqueue;

        //#############################################################################
        //! The queue empty trait.
        template<typename TQueue, typename TSfinae = void>
        struct Empty;

        //#############################################################################
        //! Queue for an accelerator
        template<typename TAcc, typename TProperty, typename TSfinae = void>
        struct QueueType;
    } // namespace traits

    //-----------------------------------------------------------------------------
    //! Queues the given task in the given queue.
    //!
    //! Special Handling for events:
    //!   If the event has previously been queued, then this call will overwrite any existing state of the event.
    //!   Any subsequent calls which examine the status of event will only examine the completion of this most recent
    //!   call to enqueue.
    template<typename TQueue, typename TTask>
    ALPAKA_FN_HOST auto enqueue(TQueue& queue, TTask&& task) -> void
    {
        traits::Enqueue<TQueue, std::decay_t<TTask>>::enqueue(queue, std::forward<TTask>(task));
    }

    //-----------------------------------------------------------------------------
    //! Tests if the queue is empty (all ops in the given queue have been completed).
    template<typename TQueue>
    ALPAKA_FN_HOST auto empty(TQueue const& queue) -> bool
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptQueue, TQueue>;
        return traits::Empty<ImplementationBase>::empty(queue);
    }

    //-----------------------------------------------------------------------------
    //! Queue based on the environment and a property
    //!
    //! \tparam TEnv Environment type, e.g.  accelerator, device or a platform.
    //!              traits::QueueType must be specialized for TEnv
    //! \tparam TProperty Property to define the behavior of TEnv.
    template<typename TEnv, typename TProperty>
    using Queue = typename traits::QueueType<TEnv, TProperty>::type;
} // namespace alpaka

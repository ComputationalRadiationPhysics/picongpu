/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/wait/Traits.hpp"

#include <type_traits>
#include <utility>

namespace alpaka
{
    struct ConceptQueue;

    //! True if TQueue is a queue, i.e. if it implements the ConceptQueue concept.
    template<typename TQueue>
    inline constexpr bool isQueue = concepts::ImplementsConcept<ConceptQueue, std::decay_t<TQueue>>::value;

    //! The queue traits.
    namespace trait
    {
        //! The queue enqueue trait.
        template<typename TQueue, typename TTask, typename TSfinae = void>
        struct Enqueue;

        //! The queue empty trait.
        template<typename TQueue, typename TSfinae = void>
        struct Empty;

        //! Queue for an accelerator
        template<typename TAcc, typename TProperty, typename TSfinae = void>
        struct QueueType;
    } // namespace trait

    //! Queues the given task in the given queue.
    //!
    //! Special Handling for events:
    //!   If the event has previously been queued, then this call will overwrite any existing state of the event.
    //!   Any subsequent calls which examine the status of event will only examine the completion of this most recent
    //!   call to enqueue.
    //!   If a queue is waiting for an event the latter's event state at the time of the API call to wait() will be
    //!   used to release the queue.
    template<typename TQueue, typename TTask>
    ALPAKA_FN_HOST auto enqueue(TQueue& queue, TTask&& task) -> void
    {
        trait::Enqueue<TQueue, std::decay_t<TTask>>::enqueue(queue, std::forward<TTask>(task));
    }

    //! Tests if the queue is empty (all ops in the given queue have been completed).
    //!
    //! \warning This function is allowed to return false negatives. An empty queue can reported as
    //! non empty because the status information are not fully propagated by the used alpaka backend.
    //! \return true queue is empty else false.
    template<typename TQueue>
    ALPAKA_FN_HOST auto empty(TQueue const& queue) -> bool
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptQueue, TQueue>;
        return trait::Empty<ImplementationBase>::empty(queue);
    }

    //! Queue based on the environment and a property
    //!
    //! \tparam TEnv Environment type, e.g.  accelerator, device or a platform.
    //!              trait::QueueType must be specialized for TEnv
    //! \tparam TProperty Property to define the behavior of TEnv.
    template<typename TEnv, typename TProperty>
    using Queue = typename trait::QueueType<TEnv, TProperty>::type;
} // namespace alpaka

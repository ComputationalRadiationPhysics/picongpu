/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"

namespace alpaka
{
    struct ConceptCurrentThreadWaitFor
    {
    };

    //! The wait traits.
    namespace trait
    {
        //! The thread wait trait.
        template<typename TAwaited, typename TSfinae = void>
        struct CurrentThreadWaitFor;

        //! The waiter wait trait.
        template<typename TWaiter, typename TAwaited, typename TSfinae = void>
        struct WaiterWaitFor;
    } // namespace trait

    //! Waits the thread for the completion of the given awaited action to complete.
    //!
    //! Special Handling for events:
    //!   If the event is re-enqueued wait() will terminate when the re-enqueued event will be ready and previously
    //!   enqueued states of the event will be ignored.
    template<typename TAwaited>
    ALPAKA_FN_HOST auto wait(TAwaited const& awaited) -> void
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptCurrentThreadWaitFor, TAwaited>;
        trait::CurrentThreadWaitFor<ImplementationBase>::currentThreadWaitFor(awaited);
    }

    //! The waiter waits for the given awaited action to complete.
    //!
    //! Special Handling if \p waiter is a queue and \p awaited an event:
    //!   The \p waiter waits for the event state to become ready based on the recently captured event state at the
    //!   time of the API call even if the event is being re-enqueued later.
    template<typename TWaiter, typename TAwaited>
    ALPAKA_FN_HOST auto wait(TWaiter& waiter, TAwaited const& awaited) -> void
    {
        trait::WaiterWaitFor<TWaiter, TAwaited>::waiterWaitFor(waiter, awaited);
    }
} // namespace alpaka

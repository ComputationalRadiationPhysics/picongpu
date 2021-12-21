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

namespace alpaka
{
    struct ConceptCurrentThreadWaitFor
    {
    };

    //-----------------------------------------------------------------------------
    //! The wait traits.
    namespace traits
    {
        //#############################################################################
        //! The thread wait trait.
        template<typename TAwaited, typename TSfinae = void>
        struct CurrentThreadWaitFor;

        //#############################################################################
        //! The waiter wait trait.
        template<typename TWaiter, typename TAwaited, typename TSfinae = void>
        struct WaiterWaitFor;
    } // namespace traits

    //-----------------------------------------------------------------------------
    //! Waits the thread for the completion of the given awaited action to complete.
    template<typename TAwaited>
    ALPAKA_FN_HOST auto wait(TAwaited const& awaited) -> void
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptCurrentThreadWaitFor, TAwaited>;
        traits::CurrentThreadWaitFor<ImplementationBase>::currentThreadWaitFor(awaited);
    }

    //-----------------------------------------------------------------------------
    //! The waiter waits for the given awaited action to complete.
    template<typename TWaiter, typename TAwaited>
    ALPAKA_FN_HOST auto wait(TWaiter& waiter, TAwaited const& awaited) -> void
    {
        traits::WaiterWaitFor<TWaiter, TAwaited>::waiterWaitFor(waiter, awaited);
    }
} // namespace alpaka

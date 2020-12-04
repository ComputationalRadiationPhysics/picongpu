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
#include <alpaka/dev/Traits.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The event management traits.
    namespace traits
    {
        //#############################################################################
        //! The event type trait.
        template<typename T, typename TSfinae = void>
        struct EventType;

        //#############################################################################
        //! The event tester trait.
        template<typename TEvent, typename TSfinae = void>
        struct IsComplete;
    } // namespace traits

    //#############################################################################
    //! The event type trait alias template to remove the ::type.
    template<typename T>
    using Event = typename traits::EventType<T>::type;

    //-----------------------------------------------------------------------------
    //! Tests if the given event has already been completed.
    template<typename TEvent>
    ALPAKA_FN_HOST auto isComplete(TEvent const& event) -> bool
    {
        return traits::IsComplete<TEvent>::isComplete(event);
    }
} // namespace alpaka

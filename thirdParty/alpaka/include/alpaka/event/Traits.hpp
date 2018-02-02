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

#include <alpaka/core/Common.hpp>

#include <alpaka/dev/Traits.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The event management specifics.
    namespace event
    {
        //-----------------------------------------------------------------------------
        //! The event management traits.
        namespace traits
        {
            //#############################################################################
            //! The event type trait.
            template<
                typename T,
                typename TSfinae = void>
            struct EventType;

            //#############################################################################
            //! The event tester trait.
            template<
                typename TEvent,
                typename TSfinae = void>
            struct Test;
        }

        //#############################################################################
        //! The event type trait alias template to remove the ::type.
        template<
            typename T>
        using Event = typename traits::EventType<T>::type;

        //-----------------------------------------------------------------------------
        //! Tests if the given event has already been completed.
        template<
            typename TEvent>
        ALPAKA_FN_HOST auto test(
            TEvent const & event)
        -> bool
        {
            return
                traits::Test<
                    TEvent>
                ::test(
                    event);
        }
    }
}

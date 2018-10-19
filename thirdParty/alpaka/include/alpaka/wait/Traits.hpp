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

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The wait specifics.
    namespace wait
    {
        //-----------------------------------------------------------------------------
        //! The wait traits.
        namespace traits
        {
            //#############################################################################
            //! The thread wait trait.
            template<
                typename TAwaited,
                typename TSfinae = void>
            struct CurrentThreadWaitFor;

            //#############################################################################
            //! The waiter wait trait.
            template<
                typename TWaiter,
                typename TAwaited,
                typename TSfinae = void>
            struct WaiterWaitFor;
        }

        //-----------------------------------------------------------------------------
        //! Waits the thread for the completion of the given awaited action to complete.
        template<
            typename TAwaited>
        ALPAKA_FN_HOST auto wait(
            TAwaited const & awaited)
        -> void
        {
            traits::CurrentThreadWaitFor<
                TAwaited>
            ::currentThreadWaitFor(
                awaited);
        }

        //-----------------------------------------------------------------------------
        //! The waiter waits for the given awaited action to complete.
        template<
            typename TWaiter,
            typename TAwaited>
        ALPAKA_FN_HOST auto wait(
            TWaiter & waiter,
            TAwaited const & awaited)
        -> void
        {
            traits::WaiterWaitFor<
                TWaiter,
                TAwaited>
            ::waiterWaitFor(
                waiter,
                awaited);
        }
    }
}

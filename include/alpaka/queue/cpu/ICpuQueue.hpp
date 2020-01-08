/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>

namespace alpaka
{
    namespace event
    {
        class EventCpu;
    }
}

namespace alpaka
{
    namespace queue
    {
        namespace cpu
        {


#if BOOST_COMP_CLANG
    // avoid diagnostic warning: "has no out-of-line virtual method definitions; its vtable will be emitted in every translation unit [-Werror,-Wweak-vtables]"
    // https://stackoverflow.com/a/29288300
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wweak-vtables"
#endif

            //#############################################################################
            //! The CPU queue interface
            class ICpuQueue
            {
            public:
                //-----------------------------------------------------------------------------
                //! enqueue the event
                virtual void enqueue(event::EventCpu &) = 0;
                //-----------------------------------------------------------------------------
                //! waiting for the event
                virtual void wait(event::EventCpu const &) = 0;
                //-----------------------------------------------------------------------------
                virtual ~ICpuQueue() = default;
            };
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
        }
    }
}

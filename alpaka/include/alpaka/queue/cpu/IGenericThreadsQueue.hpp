/* Copyright 2020 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>

namespace alpaka
{
    template<typename TDev>
    class EventGenericThreads;

#if BOOST_COMP_CLANG
// avoid diagnostic warning: "has no out-of-line virtual method definitions; its vtable will be emitted in every
// translation unit [-Werror,-Wweak-vtables]" https://stackoverflow.com/a/29288300
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#endif

    //! The CPU queue interface
    template<typename TDev>
    class IGenericThreadsQueue
    {
    public:
        //! enqueue the event
        virtual void enqueue(EventGenericThreads<TDev>&) = 0;
        //! waiting for the event
        virtual void wait(EventGenericThreads<TDev> const&) = 0;
        virtual ~IGenericThreadsQueue() = default;
    };
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
} // namespace alpaka

/* Copyright 2020 Rene Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka
{
    //! Properties to define queue behavior
    namespace property
    {
        //! The caller is waiting until the enqueued task is finished
        struct Blocking;

        //! The caller is NOT waiting until the enqueued task is finished
        struct NonBlocking;
    } // namespace property

    using namespace property;
} // namespace alpaka

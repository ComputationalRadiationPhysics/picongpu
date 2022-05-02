/* Copyright 2020 Rene Widera, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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

/* Copyright 2020 Benjamin Worpitz, René Widera, Bernhard Manfred Gruber
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#    if _OPENMP < 201307
#        error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#    endif

#    include <alpaka/dev/DevOmp5.hpp>
#    include <alpaka/queue/QueueGenericThreadsBlocking.hpp>

namespace alpaka
{
    using QueueOmp5Blocking = QueueGenericThreadsBlocking<DevOmp5>;
}

#endif

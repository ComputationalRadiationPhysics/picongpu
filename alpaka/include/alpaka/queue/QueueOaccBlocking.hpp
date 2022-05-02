/* Copyright 2020 Benjamin Worpitz, René Widera, Bernhard Manfred Gruber
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#    if _OPENACC < 201306
#        error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC 2.0 or higher!
#    endif

#    include <alpaka/dev/DevOacc.hpp>
#    include <alpaka/queue/QueueGenericThreadsBlocking.hpp>

namespace alpaka
{
    using QueueOaccBlocking = QueueGenericThreadsBlocking<DevOacc>;
}

#endif

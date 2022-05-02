/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/queue/sycl/QueueGenericSyclBase.hpp>

#    include <memory>
#    include <utility>

namespace alpaka::experimental
{
    template<typename TDev>
    using QueueGenericSyclNonBlocking = detail::QueueGenericSyclBase<TDev, false>;
}

#endif

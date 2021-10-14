/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>

namespace alpaka
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename... Ts>
    ALPAKA_FN_INLINE constexpr ALPAKA_FN_HOST_ACC void ignore_unused(Ts const&...)
    {
    }

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename... Ts>
    ALPAKA_FN_INLINE constexpr ALPAKA_FN_HOST_ACC void ignore_unused()
    {
    }

} // namespace alpaka

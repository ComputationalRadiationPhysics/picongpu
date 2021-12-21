/* Copyright 2019 Benjamin Worpitz
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
    namespace meta
    {
        template<typename T>
        struct min
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC
            constexpr auto operator()(const T& lhs, const T& rhs) const
            {
                return (lhs < rhs) ? lhs : rhs;
            }
        };

        template<typename T>
        struct max
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC
            constexpr auto operator()(const T& lhs, const T& rhs) const
            {
                return (lhs > rhs) ? lhs : rhs;
            }
        };
    } // namespace meta
} // namespace alpaka

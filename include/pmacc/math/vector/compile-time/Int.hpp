/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/math/vector/Vector.hpp"

#include <cstdint>

#include "pmacc/math/vector/compile-time/Vector.hpp"

namespace pmacc
{
    namespace math
    {
        namespace CT
        {
            /** Compile time int vector
             *
             *
             * @tparam x value for x allowed range [INT_MIN;INT_MAX-1]
             * @tparam y value for y allowed range [INT_MIN;INT_MAX-1]
             * @tparam z value for z allowed range [INT_MIN;INT_MAX-1]
             *
             * default parameter is used to distinguish between values given by
             * the user and unset values.
             */
            template<int... T_values>
            using Int = CT::Vector<std::integral_constant<int, T_values>...>;

            template<uint32_t dim, int val>
            struct make_Int;

            template<int val>
            struct make_Int<1u, val>
            {
                using type = Int<val>;
            };

            template<int val>
            struct make_Int<2u, val>
            {
                using type = Int<val, val>;
            };

            template<int val>
            struct make_Int<3u, val>
            {
                using type = Int<val, val, val>;
            };

        } // namespace CT
    } // namespace math
} // namespace pmacc

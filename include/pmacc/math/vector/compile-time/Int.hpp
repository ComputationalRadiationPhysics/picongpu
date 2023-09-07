/* Copyright 2013-2023 Heiko Burau, Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/math/vector/Vector.hpp"
#include "pmacc/traits/Limits.hpp"

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

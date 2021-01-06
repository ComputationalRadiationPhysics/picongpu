/* Copyright 2013-2021 Heiko Burau, Rene Widera
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

#include <stdint.h>
#include <boost/mpl/integral_c.hpp>
#include "pmacc/math/Vector.hpp"
#include "pmacc/traits/Limits.hpp"

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
            template<
                int x = traits::limits::Max<int>::value,
                int y = traits::limits::Max<int>::value,
                int z = traits::limits::Max<int>::value>
            struct Int : public CT::Vector<mpl::integral_c<int, x>, mpl::integral_c<int, y>, mpl::integral_c<int, z>>
            {
            };

            template<>
            struct Int<> : public CT::Vector<>
            {
            };

            template<int x>
            struct Int<x> : public CT::Vector<mpl::integral_c<int, x>>
            {
            };

            template<int x, int y>
            struct Int<x, y> : public CT::Vector<mpl::integral_c<int, x>, mpl::integral_c<int, y>>
            {
            };


            template<int dim, int val>
            struct make_Int;

            template<int val>
            struct make_Int<1, val>
            {
                using type = Int<val>;
            };

            template<int val>
            struct make_Int<2, val>
            {
                using type = Int<val, val>;
            };

            template<int val>
            struct make_Int<3, val>
            {
                using type = Int<val, val, val>;
            };

        } // namespace CT
    } // namespace math
} // namespace pmacc

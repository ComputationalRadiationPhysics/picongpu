/* Copyright 2013-2021 Heiko Burau, Rene Widera, Axel Huebl
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
#include "pmacc/math/Vector.hpp"
#include <boost/mpl/integral_c.hpp>
#include "pmacc/traits/Limits.hpp"

namespace pmacc
{
    namespace math
    {
        namespace CT
        {
            /** Compile time uint64_t vector
             *
             *
             * @tparam x value for x allowed range [0;max uint64_t value -1]
             * @tparam y value for y allowed range [0;max uint64_t value -1]
             * @tparam z value for z allowed range [0;max uint64_t value -1]
             *
             * default parameter is used to distinguish between values given by
             * the user and unset values.
             */
            template<
                uint64_t x = traits::limits::Max<uint64_t>::value,
                uint64_t y = traits::limits::Max<uint64_t>::value,
                uint64_t z = traits::limits::Max<uint64_t>::value>
            struct UInt64
                : public CT::
                      Vector<mpl::integral_c<uint64_t, x>, mpl::integral_c<uint64_t, y>, mpl::integral_c<uint64_t, z>>
            {
            };

            template<>
            struct UInt64<> : public CT::Vector<>
            {
            };

            template<uint64_t x>
            struct UInt64<x> : public CT::Vector<mpl::integral_c<uint64_t, x>>
            {
            };

            template<uint64_t x, uint64_t y>
            struct UInt64<x, y> : public CT::Vector<mpl::integral_c<uint64_t, x>, mpl::integral_c<uint64_t, y>>
            {
            };


        } // namespace CT
    } // namespace math
} // namespace pmacc

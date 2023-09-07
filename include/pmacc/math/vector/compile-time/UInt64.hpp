/* Copyright 2013-2023 Heiko Burau, Rene Widera, Axel Huebl
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
            template<uint64_t... T_values>
            using UInt64 = CT::Vector<std::integral_constant<uint64_t, T_values>...>;
        } // namespace CT
    } // namespace math
} // namespace pmacc

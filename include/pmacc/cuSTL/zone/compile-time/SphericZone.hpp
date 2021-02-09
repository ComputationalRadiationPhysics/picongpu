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

#include "pmacc/math/Vector.hpp"

namespace pmacc
{
    namespace zone
    {
        namespace CT
        {
            /* spheric (no holes), cartesian, compile-time zone
             *
             * \tparam _Size compile-time vector (pmacc::math::CT::Size_t) of the zone's size.
             * \tparam _Offset compile-time vector (pmacc::math::CT::Size_t) of the zone's offset. default is a zero
             * vector.
             *
             * This is a zone which is simply described by a size and a offset.
             *
             * Compile-time version of zone::SphericZone
             *
             */
            template<typename _Size, typename _Offset = typename math::CT::make_Int<_Size::dim, 0>::type>
            struct SphericZone
            {
                typedef _Size Size;
                typedef _Offset Offset;
                static constexpr int dim = Size::dim;
            };

        } // namespace CT
    } // namespace zone
} // namespace pmacc

/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include <pmacc/types.hpp>
#include <pmacc/math/vector/Int.hpp>

namespace picongpu
{
    template<uint32_t T_support>
    struct ShiftCoordinateSystemNative
    {
        /**shift to new coordinat system
         *
         * shift cursor and vector to new coordinate system
         * @param curser curser to memory
         * @param vector short vector with coordinates in old system
         * @param fieldPos vector with relative coordinates for shift ( value range [0.0;0.5] )
         */
        template<typename Cursor, typename Vector>
        HDINLINE void operator()(Cursor& cursor, Vector& vector, const floatD_X& fieldPos)
        {
            for(uint32_t i = 0; i < simDim; ++i)
                vector[i] -= fieldPos[i];
        }
    };

} // namespace picongpu

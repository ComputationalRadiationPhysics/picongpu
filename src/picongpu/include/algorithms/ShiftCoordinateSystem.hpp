/**
 * Copyright 2013 Axel Huebl, Heiko Burau, René Widera
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

#include "types.h"
#include "types.h"
#include "math/vector/Int.hpp"

namespace picongpu
{

struct ShiftCoordinateSystem
{

    /**shift to new coordinat system
     * 
     * shift cursor and vector to new coordinate system
     * @param curser curser to memory
     * @param vector short vector with coordinates in old system
     * @param fieldPos vector with relative coordinates for shift ( value range [0.0;0.5] )
     */
    template<typename Cursor, typename Vector >
        HDINLINE void operator()(Cursor& cursor, Vector& vector, const float3_X & fieldPos)
    {

        if (speciesParticleShape::ParticleShape::support % 2 == 0)
        {
            //even support

            /* for any direction
             * if fieldPos == 0.5 and vector<0.5 
             * shift curser+(-1) and new_vector=old_vector-(-1)
             * 
             * (vector.x() < fieldPos.x()) is equal ((fieldPos == 0.5) && (vector<0.5))
             */
            float3_X coordinate_shift(
                                      -float_X(vector.x() < fieldPos.x()),
                                      -float_X(vector.y() < fieldPos.y()),
                                      -float_X(vector.z() < fieldPos.z())
                                      );
            cursor = cursor(
                            PMacc::math::Int < 3 > (
                                               coordinate_shift.x(),
                                               coordinate_shift.y(),
                                               coordinate_shift.z()
                                               ));
            //same as: vector = vector - fieldPos - coordinate_shift;
            vector -= (fieldPos + coordinate_shift);
        }
        else
        {
            //odd support

            /* for any direction
             * if fieldPos < 0.5 and vector> 0.5 
             * shift curser+1 and new_vector=old_vector-1
             */
            float3_X coordinate_shift(
                                      float_X(vector.x() > float_X(0.5) && fieldPos.x() != float_X(0.5)),
                                      float_X(vector.y() > float_X(0.5) && fieldPos.y() != float_X(0.5)),
                                      float_X(vector.z() > float_X(0.5) && fieldPos.z() != float_X(0.5))
                                      );
            cursor = cursor(
                            PMacc::math::Int < 3 > (
                                               coordinate_shift.x(),
                                               coordinate_shift.y(),
                                               coordinate_shift.z()
                                               ));
            //same as: vector = vector - fieldPos - coordinate_shift;
            vector -= (fieldPos + coordinate_shift);

        }






    }
};

} // picongpu

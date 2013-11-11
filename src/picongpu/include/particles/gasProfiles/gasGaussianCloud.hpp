/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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
#include "simulation_defines.hpp"

namespace picongpu
{
    namespace gasGaussianCloud
    {

        /** Calculate the gas density, divided by the maximum density GAS_DENSITY
         * 
         * @param y as distance in propagation direction (unit: meters / UNIT_LENGTH)
         * @return float_X between 0.0 and 1.0
         */
        DINLINE float_X calcNormedDensitiy( float3_X pos )
        {
            if (pos.y() < VACUUM_Y) return float_X(0.0);

            const float3_X exponent = float3_X( math::abs((pos.x() - GAS_CENTER_X)/GAS_SIGMA_X),
                                                 math::abs((pos.y() - GAS_CENTER_Y)/GAS_SIGMA_Y),
                                                 math::abs((pos.z() - GAS_CENTER_Z)/GAS_SIGMA_Z) );

            const float_X density = math::exp(GAS_FACTOR * __powf(exponent.x(), GAS_POWER))
                                * math::exp(GAS_FACTOR * __powf(exponent.y(), GAS_POWER))
                                * math::exp(GAS_FACTOR * __powf(exponent.z(), GAS_POWER));
            return density;
        }
    }
}






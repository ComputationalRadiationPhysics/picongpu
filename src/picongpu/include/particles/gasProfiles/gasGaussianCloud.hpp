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
DINLINE float_X calcNormedDensitiy(floatD_X pos)
{
    if (pos.y() < VACUUM_Y) return float_X(0.0);

    floatD_X exponent  = float_X(math::abs((pos - GAS_CENTER) / GAS_SIGMA));


    float_X density=1; 
    for(uint32_t d=0;d<simDim;++d)
    density *= math::exp(GAS_FACTOR * __powf(exponent[d], GAS_POWER));
    
    return density;
}
} //namespace gasGaussianCloud
} //namespace picongpu






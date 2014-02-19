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
    namespace gasLinExp
    {

        /** Calculate the gas density, divided by the maximum density GAS_DENSITY
         * 
         * @param pos as 3D length vector offset to global left top front cell
         * @return float_X between 0.0 and 1.0
         */
        DINLINE float_X calcNormedDensitiy( floatD_X pos )
        {
            if (pos.y() < VACUUM_Y) return float_X(0.0);

            float_X density = float_X(0.0);
            
            if (pos.y() <= GAS_Y_MAX) // linear slope
                density = GAS_A * pos.y() + GAS_B;
            else // exponential slope
                density = math::exp( (pos.y() - GAS_Y_MAX) * GAS_D );
            
            // avoid < 0 densities for the linear slope
            if (density < float_X(0.0))
                density = float_X(0.0);
            
            return density;
        }
    } // namespace gasLinExp
} // namespace picongpu

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
 


#ifndef GASGAUSSIAN_HPP
#define	GASGAUSSIAN_HPP

#include "types.h"
#include "simulation_defines.hpp"

namespace picongpu
{
    namespace gasGaussian
    {

        /** Calculate the gas density, divided by the maximum density GAS_DENSITY
         * 
         * @param pos as 3D length vector offset to global left top front cell
         * @return float_X between 0.0 and 1.0
         */
        DINLINE float_X calcNormedDensitiy( floatD_X pos )
        {
            if (pos.y() < VACUUM_Y) return float_X(0.0);

            float_X exponent = float_X(0.0);
            if (pos.y() < GAS_CENTER_LEFT)
                exponent = math::abs((pos.y() - GAS_CENTER_LEFT) / GAS_SIGMA_LEFT);
            else if (pos.y() > GAS_CENTER_RIGHT)
                exponent = math::abs((pos.y() - GAS_CENTER_RIGHT) / GAS_SIGMA_RIGHT);

            const float_X density = math::exp(float_X(GAS_FACTOR) * __powf(exponent, GAS_POWER));
            return density;
        }
    }
}

#endif	/* GASGAUSSIAN_HPP */




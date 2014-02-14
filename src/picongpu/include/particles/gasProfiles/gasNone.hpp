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
 


#ifndef GASNONE_HPP
#define	GASNONE_HPP

#include "types.h"
#include "simulation_defines.hpp"

namespace picongpu
{
    namespace gasNone
    {

        /** Calculate the gas density, divided by the maximum density GAS_DENSITY
         * 
         * @return float_X between 0.0 and 1.0
         */
        DINLINE float_X calcNormedDensitiy( floatD_X  )
        {

            return float_X(0.0);

        }
    }
}

#endif	/* GASNONE_HPP */




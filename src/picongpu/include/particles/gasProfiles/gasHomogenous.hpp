/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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
 


#ifndef GASHOMOGENOUS_HPP
#define	GASHOMOGENOUS_HPP

#include "types.h"
#include "simulation_defines.hpp"
#include "memory/buffers/GridBuffer.hpp"

namespace picongpu
{
    namespace gasHomogenous
    {
        template<class Type>
        bool gasSetup(GridBuffer<Type, simDim> &fieldBuffer)
        {
            return true;
        }

        /** Calculate the gas density, divided by the maximum density GAS_DENSITY
         * 
         * @param pos as 3D length vector offset to global left top front cell
         * @return float_X between 0.0 and 1.0
         */
        template<unsigned DIM, typename FieldBox>
        DINLINE float_X calcNormedDensity( floatD_X pos, const DataSpace<DIM>&, FieldBox )
        {
            if (pos.y() < VACUUM_Y
                || pos.y() >= (GAS_LENGTH + VACUUM_Y)) return float_X(0.0);

            return float_X(1.0);
        }
    }
}

#endif	/* GASHOMOGENOUS_HPP */

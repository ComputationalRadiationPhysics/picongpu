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

#include <string>

#include "types.h"
#include "simulation_defines.hpp"

namespace picongpu
{

    struct IonMethods
    {
    public:
        typedef float_X MassType;
        typedef float_X ChargeType;
        
        typedef typename MappingDesc::SuperCellSize SuperCellSize;

        DINLINE float_X getM0_2(const float_X weighting)
        {
            return (M_ION * M_ION * weighting * weighting);
        }

        DINLINE MassType getMass(const float_X weighting)
        {
            return (M_ION * weighting);
        }

        DINLINE ChargeType getCharge(const float_X weighting)
        {
            return (Q_ION * weighting);
        }
        
        HINLINE static std::string getName( )
        {
            return "i";
        }

        enum
        {
            CommunicationTag = PAR_IONS
        };

    };

}//namespace picongpu

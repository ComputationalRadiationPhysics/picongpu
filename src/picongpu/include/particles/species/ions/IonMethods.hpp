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
        typedef uint32_t ChargeStateType;
        
        typedef typename MappingDesc::SuperCellSize SuperCellSize;

        HDINLINE static float_X getM0_2(const float_X weighting)
        {
            return (M_ION * M_ION * weighting * weighting);
        }

        HDINLINE static MassType getMass(const float_X weighting)
        {
            return (M_ION * weighting);
        }

        HDINLINE static ChargeType getCharge(const float_X weighting, uint32_t chargeState)
        {
            return (Q_ION * weighting * chargeState);
        }
        /* return charge state of the ion*/
        /* HDINLINE static ChargeStateType getChargeState(const float_X weighting, uint32_t chargeState)
         * {
         *    return (chargeState * weighting);
         * }
         */
        enum
        {
            CommunicationTag = PAR_IONS
        };

    };

}//namespace picongpu

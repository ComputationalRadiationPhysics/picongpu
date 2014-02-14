/**
 * Copyright 2013 Axel Huebl
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
    namespace gasFreeFormula
    {

        /** Calculate the gas density, divided by the maximum density GAS_DENSITY
         * 
         * @param pos as 3D length vector offset to global left top front cell
         * @return float_X between 0.0 and 1.0
         */
        DINLINE float_X calcNormedDensitiy( floatD_X pos)
        {
            if (pos.y() < VACUUM_Y) return float_X(0.0);

            float_64 _unit_length=UNIT_LENGTH;
            const floatD_64 pos_SI = typeCast<float_64>( pos ) * _unit_length;

            /* expected return value of the profile: [0.0:1.0] */
            SI::GasProfile gasProfile;
            const float_64 density_dbl = gasProfile( pos_SI );

            float_X density = typeCast<float_X>( density_dbl );

            /* validate formula and clip to [0.0:1.0] */
            density *= typeCast<float_X>( density >= 0.0 );
            if( density > 1.0 ) density = 1.0;

            return density;
        }
    }
}

/**
 * Copyright 2013, 2015 Heiko Burau, Rene Widera, Richard Pausch
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
#include "plugins/radiation/parameters.hpp"

namespace radiation
{
using namespace PMacc;
using namespace picongpu;

template<bool hasRadFlag>
struct PushExtension
{

    template<typename MomType, typename MassType >
        HDINLINE void operator()(
                                MomType& mom_mt1,
                                const MomType & mom,
                                const MassType mass,
                                bool& radiationFlag
                                )
    {
        bool radFlag = radiationFlag;
#if (RAD_ACTIVATE_GAMMA_FILTER==1)
        if (!radFlag)
        {
            const float_X c2 = SPEED_OF_LIGHT*SPEED_OF_LIGHT;
            // Radiation marks only a particle if it has a high velocity
            // marked particle means that momentumPrev1 is not 0.0 in one direction

            const float_X abs2_mom = abs2(mom);
            if (((abs2_mom)>((parameters::RadiationGamma * parameters::RadiationGamma - float_X(1.0)) * mass * mass * c2)))
            {
                radFlag = true;
                radiationFlag = true;
            }
        }
#endif
        /*\todo: is it faster if all run memcpy and we have no if condition?*/
        if (radFlag)
            mom_mt1 = mom;
        /* end radiation */
    }
};

template<>
struct PushExtension < false >
{

    template<typename MomType, typename MassType >
        HDINLINE void operator()(
                                MomType& mom_mt1,
                                const MomType & mom,
                                const MassType mass
                                )
    {
        mom_mt1 = mom;

    }
};

}




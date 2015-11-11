/**
 * Copyright 2013-2015 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Richard Pausch
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

namespace picongpu
{
    namespace particlePusherFree
    {
        template<class Velocity, class Gamma>
        struct Push
        {

            template<typename T_FunctorFieldE, typename T_FunctorFieldB, typename T_Pos, typename T_Mom, typename T_Mass,
                     typename T_Charge, typename T_Weighting>
                    __host__ DINLINE void operator()(
                                                        const T_FunctorFieldB,
                                                        const T_FunctorFieldE,
                                                        T_Pos& pos,
                                                        T_Mom& mom,
                                                        const T_Mass mass,
                                                        const T_Charge,
                                                        const T_Weighting)
            {
                typedef T_Mom MomType;

                Velocity velocity;
                const MomType vel = velocity(mom, mass);


                for(uint32_t d=0;d<simDim;++d)
                {
                    pos[d] += (vel[d] * DELTA_T) / cellSize[d];
                }
            }
        };
    } //namespace
}

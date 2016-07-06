/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Alexander Grund, Richard Pausch
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

#include "pmacc_types.hpp"

namespace picongpu
{
    namespace particlePusherPhoton
    {
        template<class Velocity, class Gamma>
        struct Push
        {
            /* this is an optional extension for sub-sampling pushes that enables grid to particle interpolation
             * for particle positions outside the super cell in one push
             */
            typedef typename PMacc::math::CT::make_Int<simDim,0>::type LowerMargin;
            typedef typename PMacc::math::CT::make_Int<simDim,0>::type UpperMargin;

            template<typename T_FunctorFieldE, typename T_FunctorFieldB, typename T_Pos, typename T_Mom, typename T_Mass,
                     typename T_Charge,  typename T_Weighting>
                    __host__ DINLINE void operator()(
                                                        const T_FunctorFieldB,
                                                        const T_FunctorFieldE,
                                                        T_Pos& pos,
                                                        T_Mom& mom,
                                                        const T_Mass,
                                                        const T_Charge,
                                                        const T_Weighting)
            {
                typedef T_Mom MomType;

                const float_X mom_abs = math::abs( mom );
                const MomType vel = mom * ( SPEED_OF_LIGHT / mom_abs );

                for(uint32_t d=0;d<simDim;++d)
                {
                    pos[d] += (vel[d] * DELTA_T) / cellSize[d];
                }
            }

            static PMacc::traits::StringProperty getStringProperties()
            {
                PMacc::traits::StringProperty propList( "name", "other" );
                propList["param"] = "free streaming photon pusher";
                return propList;
            }
        };
    } //namespace
}

/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera,
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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"


namespace picongpu
{
    namespace particlePusherFree
    {
        template<class Velocity, class Gamma>
        struct Push
        {
            /* this is an optional extension for sub-sampling pushes that enables grid to particle interpolation
             * for particle positions outside the super cell in one push
             */
            using LowerMargin = typename pmacc::math::CT::make_Int<simDim,0>::type;
            using UpperMargin = typename pmacc::math::CT::make_Int<simDim,0>::type;

            template< typename T_FunctorFieldE, typename T_FunctorFieldB, typename T_Particle, typename T_Pos >
            HDINLINE void operator()(
                const T_FunctorFieldB functorBField,
                const T_FunctorFieldE functorEField,
                T_Particle & particle,
                T_Pos & pos,
                const uint32_t
            )
            {
                float_X const weighting = particle[ weighting_ ];
                float_X const mass = attribute::getMass( weighting, particle );

                using MomType = momentum::type;
                MomType const mom = particle[ momentum_ ];

                Velocity velocity;
                const MomType vel = velocity(mom, mass);


                for(uint32_t d=0;d<simDim;++d)
                {
                    pos[d] += (vel[d] * DELTA_T) / cellSize[d];
                }
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList( "name", "other" );
                propList["param"] = "free streaming";
                return propList;
            }
        };
    } //namespace
}

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
 


#ifndef PARTICLEPUSHERFREE_HPP
#define	PARTICLEPUSHERFREE_HPP

#include "types.h"

namespace picongpu
{
    namespace particlePusherFree
    {
        template<class Velocity, class Gamma>
        struct Push
        {

            template<typename EType, typename BType, typename PosType, typename MomType, typename MassType, typename ChargeType >
                    __host__ DINLINE void operator()(
                                                        const BType bField,
                                                        const EType eField,
                                                        PosType& pos,
                                                        MomType& mom,
                                                        const MassType mass,
                                                        const ChargeType charge)
            {

                Velocity velocity;
                const PosType vel = velocity(mom, mass);

                /* IMPORTANT: 
                 * use float_X(1.0)+X-float_X(1.0) because the rounding of float_X can create position from [-float_X(1.0),2.f],
                 * this breaks ower definition that after position change (if statements later) the position must [float_X(0.0),float_X(1.0))
                 * 1.e-9+float_X(1.0) = float_X(1.0) (this is not allowed!!!
                 * 
                 * If we don't use this fermi crash in this kernel in the time step n+1 in field interpolation
                 */
                pos.x() += float_X(1.0) + (vel.x() * DELTA_T / CELL_WIDTH);
                pos.y() += float_X(1.0) + (vel.y() * DELTA_T / CELL_HEIGHT);
                pos.z() += float_X(1.0) + (vel.z() * DELTA_T / CELL_DEPTH);

                pos.x() -= float_X(1.0);
                pos.y() -= float_X(1.0);
                pos.z() -= float_X(1.0);

            }
        };
    } //namespace
}

#endif	/* PARTICLEPUSHERFREE_HPP */


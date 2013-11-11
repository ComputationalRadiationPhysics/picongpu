/**
 * Copyright 2013 Axel Huebl, Heiko Burau, René Widera, Richard Pausch
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
 


#ifndef PARTICLEPUSHERVAY_HPP
#define	PARTICLEPUSHERVAY_HPP

#include "types.h"
#include "simulation_defines.hpp"

namespace picongpu
{
namespace particlePusherVay
{

template<class Velocity, class Gamma>
struct Push
{

    template<typename EType, typename BType, typename PosType, typename MomType, typename MassType, typename ChargeType >
        __host__ DINLINE void operator()(
                                            const BType bField, /* at t=0 */
                                            const EType eField, /* at t=0 */
                                            PosType& pos, /* at t=0 */
                                            MomType& mom, /* at t=-1/2 */
                                            const MassType mass,
                                            const ChargeType charge)
    {

        /*   
             time index in paper is reduced by a half: i=0 --> i=-1/2 so that momenta are 
             at half time steps and fields and locations are at full time steps

     Here the real (PIConGPU) momentum (p) is used, not the momentum from the Vay paper (u)
     p = m_0 * u
         */
        const float_X deltaT = DELTA_T;
        const float_X factor = 0.5 * charge * deltaT;
        Gamma gamma;
        Velocity velocity;

        // first step in Vay paper:
        const float3_X velocity_atMinusHalf = velocity(mom, mass);
        //mom /(mass*mass + abs2(mom)/(SPEED_OF_LIGHT*SPEED_OF_LIGHT));
        const MomType momentum_atZero = mom + factor * (eField + math::cross(velocity_atMinusHalf, bField));

        // second step in Vay paper:
        const MomType momentum_prime = momentum_atZero + factor * eField;
        const float_X gamma_prime = gamma(momentum_prime, mass);
        //sqrtf(1.0 + abs2(momentum_prime*(1.0/(mass * SPEED_OF_LIGHT))));
        const sqrt_Vay::float3_X tau = factor / mass * bField;
        const sqrt_Vay::float_X u_star = math::dot( typeCast<sqrt_Vay::float_X>(momentum_prime), tau ) / typeCast<sqrt_Vay::float_X>( SPEED_OF_LIGHT * mass );
        const sqrt_Vay::float_X sigma = gamma_prime * gamma_prime - math::abs2( tau );
        const sqrt_Vay::float_X gamma_atPlusHalf = math::sqrt( sqrt_Vay::float_X(0.5) *
            ( sigma +
              math::sqrt( sigma * sigma +
                          sqrt_Vay::float_X(4.0) * ( math::abs2( tau ) + u_star * u_star ) )
            )
                                                    );
        const float3_X t = tau * (float_X(1.0) / gamma_atPlusHalf);
        const float_X s = float_X(1.0) / (float_X(1.0) + math::abs2(t));
        const MomType momentum_atPlusHalf = s * (momentum_prime + math::dot(momentum_prime, t) * t + math::cross(momentum_prime, t));

        mom = momentum_atPlusHalf;

        const float3_X vel = velocity(momentum_atPlusHalf, mass);

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

#endif	/* PARTICLEPUSHERVAY_HPP */


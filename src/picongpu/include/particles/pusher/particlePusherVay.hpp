/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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
#include "simulation_defines.hpp"

namespace picongpu
{
namespace particlePusherVay
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
             typename T_Charge, typename T_Weighting >
        __host__ DINLINE void operator()(
                                            const T_FunctorFieldB functorBField, /* at t=0 */
                                            const T_FunctorFieldE functorEField, /* at t=0 */
                                            T_Pos& pos, /* at t=0 */
                                            T_Mom& mom, /* at t=-1/2 */
                                            const T_Mass mass,
                                            const T_Charge charge,
                                            const  T_Weighting)
    {
        typedef T_Mom MomType;

        PMACC_AUTO( bField , functorBField(pos));
        PMACC_AUTO( eField , functorEField(pos));
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
        //algorithms::math::sqrt(1.0 + abs2(momentum_prime*(1.0/(mass * SPEED_OF_LIGHT))));
        const sqrt_Vay::float3_X tau(factor / mass * bField);
        const sqrt_Vay::float_X u_star = math::dot( precisionCast<sqrt_Vay::float_X>(momentum_prime), tau ) / precisionCast<sqrt_Vay::float_X>( SPEED_OF_LIGHT * mass );
        const sqrt_Vay::float_X sigma = gamma_prime * gamma_prime - math::abs2( tau );
        const sqrt_Vay::float_X gamma_atPlusHalf = math::sqrt( sqrt_Vay::float_X(0.5) *
            ( sigma +
              math::sqrt( sigma * sigma +
                          sqrt_Vay::float_X(4.0) * ( math::abs2( tau ) + u_star * u_star ) )
            )
                                                    );
        const float3_X t(tau * (float_X(1.0) / gamma_atPlusHalf));
        const float_X s = float_X(1.0) / (float_X(1.0) + math::abs2(t));
        const MomType momentum_atPlusHalf = s * (momentum_prime + math::dot(momentum_prime, t) * t + math::cross(momentum_prime, t));

        mom = momentum_atPlusHalf;

        const float3_X vel = velocity(momentum_atPlusHalf, mass);

        for(uint32_t d=0;d<simDim;++d)
        {
            pos[d] += (vel[d] * DELTA_T) / cellSize[d];
        }
    }
};
} //namespace particlePusherVay
} //namespace picongpu

/* Copyright 2013-2020 Heiko Burau, Rene Widera, Richard Pausch, Annegret Roeszler, Klaus Steiniger
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
#include "picongpu/traits/attribute/GetCharge.hpp"

namespace picongpu
{
namespace particlePusherHC
{
/* Implementation of the Higuera-Cary pusher as presented in doi:10.1063/1.4979989.
 * A correction is applied to the given formulas as documented by the WarpX team:
 * (https://github.com/ECP-WarpX/WarpX/issues/320).
 * 
 * Note, while Higuera and Ripperda present the formulas for the quantity u = gamma * v,
 * PIConGPU uses the real momentum p = gamma * m * v = u * m for calculations.
 * 
 * Further references:
 * [Higuera's article on arxiv](https://arxiv.org/abs/1701.05605)
 * [Riperda's comparison of relativistic particle integrators](https://doi.org/10.3847/1538-4365/aab114)
 */

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
        float_X const mass = attribute::getMass( weighting , particle );
        float_X const charge = attribute::getCharge( weighting , particle );

        using MomType = momentum::type;
        MomType const mom = particle[ momentum_ ];

        auto bField  = functorBField(pos);
        auto eField  = functorEField(pos);

        const float_X deltaT = DELTA_T;


        Gamma gamma;

        /*
         * Momentum update
         * Notation is according to Ripperda's paper
         */
        // First half electric field acceleration
        const MomType mom_minus = mom + float_X(0.5) * charge * eField * deltaT;

        // Auxiliary quantitites
        const sqrt_HC::float_X gamma_minus = gamma( mom_minus , mass );

        const sqrt_HC::float3_X tau = precisionCast<sqrt_HC::float_X>( float_X(0.5) * bField * charge * deltaT / mass );

        const sqrt_HC::float_X sigma = pmacc::math::abs2( gamma_minus ) - pmacc::math::abs2( tau );

        const sqrt_HC::float_X u_star = pmacc::math::dot( precisionCast<sqrt_HC::float_X>( mom_minus ), tau ) / precisionCast<sqrt_HC::float_X>( mass * SPEED_OF_LIGHT );

        const sqrt_HC::float_X gamma_plus = math::sqrt( 
                sqrt_HC::float_X(0.5) * ( sigma + math::sqrt( 
                        pmacc::math::abs2( sigma ) + sqrt_HC::float_X(4.0) * ( pmacc::math::abs2( tau ) + pmacc::math::abs2( u_star ) )
                    ) )
            );

        const sqrt_HC::float3_X t_vector =  tau / gamma_plus;

        const sqrt_HC::float_X s = sqrt_HC::float_X(1.0) / ( sqrt_HC::float_X(1.0) + pmacc::math::abs2( t_vector ) );

        // Rotation step
        const MomType mom_plus = precisionCast<float_X>( s * (
                precisionCast<sqrt_HC::float_X>( mom_minus )
                + pmacc::math::dot( precisionCast<sqrt_HC::float_X>( mom_minus ) , t_vector ) * t_vector
                + pmacc::math::cross( precisionCast<sqrt_HC::float_X>( mom_minus ) , t_vector) 
            ) );

        // Second half electric field acceleration (Note correction mom_minus -> mom_plus compared to Ripperda)
        const  sqrt_HC::float3_X mom_diff = sqrt_HC::float_X(0.5) * precisionCast<sqrt_HC::float_X>( eField * charge * deltaT ) 
            + pmacc::math::cross( precisionCast<sqrt_HC::float_X>( mom_plus ) , t_vector );
        
        const MomType new_mom = mom_plus + precisionCast<float_X>( mom_diff );

        particle[ momentum_ ] = new_mom;

        /*
         * Position update
         */
        Velocity velocity;

        const float3_X vel = velocity( new_mom , mass );

        for( uint32_t d=0 ; d<simDim ; ++d )
        {
            pos[d] += ( vel[d] * deltaT ) / cellSize[d];
        }
    }

    static pmacc::traits::StringProperty getStringProperties()
    {
        pmacc::traits::StringProperty propList( "name", "HC" );
        return propList;
    }
};
} // namespace particlePusherHC
} // namespace picongpu

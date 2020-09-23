/* Copyright 2020 Brian Marre
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

/** @file 
 */

#pragma once

#include <pmacc/algorithms/math.hpp>
#include <utility>


namespace picongpu
{
namespace particles
{
namespace atomicPhysics
{
namespace electronDistribution
{
namespace histogram2
{

    template<
        uint32_t T_orderApprox,     // error approximation of order 2 * T_orderApprox +1
                                    // in crosssection + velocity and 0th order in electron density
        uint32_t T_numSamplePoints, // number of sample points used for numerical sigma differentiation
        typename T_WeightingGen     // type of numerical differentiation method used
    >
    class RateRelativeError
    {
        // necessary for velocity derivation
        float_X mass;

        //storage of weights for later use
        float_X weights[ T_numSamplePoints * ( 2u + T_orderApprox + 2u ) ];

        /** returns the k-th of T_numNodes chebyshev nodes x_k, interval [-1, 1]
        *
        * x_k = cos( (2k-1)/(2n) * pi )
        *
        * @tparam T_numNodes ... how many chebyshev nodes are requested
        * @param k ... index of chebyshev Nodes
        *
        * BEWARE: k=0 is NOT allowed, k \in {1, ... , T_numSamplePoints}
        * BEWARE: max({ x_k }) = x_1 and min({ x_k }) = x_(T_numSamplePoints)
        *
        * see https://en.wikipedia.org/wiki/Chebyshev_nodes for more information
        */
        template< uint32_t T_numNodes
        >
        DINLINE static float_X chebyshevNodes(
            uint32_t const k
        )
        {
            // check for bounds on k
            // return boundaries if outside of boundary
            if ( k < 1u )
                return -1._X;
            if ( k > T_numNodes )
                return 1._X;

            return static_cast< float_X >(
                pmacc::algorithms::math::cos< float_X >(
                    (2u*k - 1u)_X/( 2 * T_numNodes )_X *
                    pmacc::algorithms::math::Pi::value
                    )
                );
        }

    public:

        DINLINE void init( float_X mass )
        {
            this->mass = mass;

            float_X samplePoints[ T_numSamplePoints ];

            // include reference point
            samplePoints[ 0 ] = 0._X;

            for ( uint32_t i = 1u; i < T_numSamplePoints; i++ )
            {
                samplePoints[ i ] = chebyshevNodes< ( T_numSamplePoints - 1u ) >( i + 1u );
            }

            // calculate weightings for each order derivative until maximum
            for ( uint32_t i = 0u; i <= (2u * T_orderApprox + 1u); i++ )    // i order of derivation
            {
                for (uint32_t j = 0u; j < T_numSamplePoints; j++ )  // j sample point
                {
                    this->weights[ j + i * T_numSamplePoints ] = T_WeightingGen::weighting<
                    T_numSamplePoints,
                    j
                >(
                    i,
                    samplePoints
                    );
                }
            }
        }


        // TODO:
        DINLINE float_X operator() ( float_X dE, float_X E, float_X mass ) // TODO: get acess to sigma values
        {
            float_X result = 0;

            for ( uint32_t a = 0u; a <= T_orderApprox; a++)
            {
                for ( uint32_t o = 0u; o <= 2* T_orderApprox + 1u; o ++ )
                    {
                        result += 1._X/( fak( o ) * fak( 2u*a+1u - o ) ) *
                            sigmaDerivative( E, dE, o, 0._X ) * // sigma
                            velocityDerivative( E, 2u*a+1u - o, this->mass ) *
                            1._X/( 2u*a + 2u) *
                            pmacc::algorithms::math::pow(
                                dE/2._X,
                                static_cast< int >( 2u*a + 2u)
                                ) *
                            2._X;
                    }
            }
        }

    private:
        DINLINE static uint32_t fak ( const uint32_t n )
        {
            uint32_t result = 1u;

            for ( uint32_t i = 1u; i <= n; i++ )
            {
                result *= i;
            }

            return result;
        }

        DINLINE static float_X velocityDerivative(
            float_X E,
            uint32_t Z,
            float_X mass
            )
        {
            float_X result;

            result = pmacc::algorithms::math::pow( 2._X/mass, 1._X/2._X )
            * pmacc::algorithms::math::sqrt( E )
            / pmacc::algorithms::math::pow(
                E,
                static_cast< int >( Z )
                );

            for ( uint32_t i = 0u; i <= static_cast< int >( Z ) - 1; i++ )
            {
                result *= ( 1._X/2._X - i );
            }
        }

        // TODO: add access to sigma
        DINLINE float_X sigmaDerivative( float_X E, float_X dE, uint32_t n )
        {
            float_X result = 0._X;

            for( uint32_t j = 0u; j < T_numSamplePoints; j++ )
            {
                result += this->weights[ n  * T_numSamplePoints + j ]
                    * 1._X / // sigma( chebyshevNodes(  ) )
                    dE;
            }

            return result;
        }
    };

} // namespace histogram2
} // namespace electronDistribution
} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu

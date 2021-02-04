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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/particles/atomicPhysics/AtomicRate.hpp"

#include <utility>
#include <pmacc/algorithms/math.hpp>


#pragma once

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

namespace mathFunc = pmacc::algorithms::math;

    template<
        uint32_t T_minOrderApprox,  // half of starting order of error approximation,
                                    // in crosssection + velocity and 0th order in
                                    // electron density
        uint32_t T_maxOrderApprox,  // half of maximum order of error approximation,
                                    // T_minOrderapprox <= a <= T_maxOrderApprox,
                                    //  2*a ... order of a term of the rate approximation
        uint32_t T_numSamplePoints, // number of sample points used for numerical differentiation
                                    // >= 2*a + 1
        typename T_WeightingGen,    // type of numerical differentiation method used
        typename T_AtomicRate,      // atomic rate functor
        typename T_AtomicDataBox,
        typename T_ConfigNumberDataType
    >
    class RateRelativeError
    {
        // TODO: add pmacc assert to check T_numSamplePoints >= 2*T_maxOrderApprox + 1
        using AtomicRate = T_AtomicRate;
        // necessary for velocity derivation

        // storage of weights for later use
        float_X weights[ T_numSamplePoints * (2u * T_maxOrderApprox + 1u) ];

        // return unitless
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
        template< uint32_t T_numNodes >
        DINLINE static float_X chebyshevNodes( uint32_t const k )
        {
            // check for bounds on k
            // return boundaries if outside of boundary
            if ( k < 1u )
                return -1._X;
            if ( k > T_numNodes )
                return 1._X;

            return pmacc::algorithms::math::cos< float_X >(
                static_cast< float_X >(2u*k - 1u) /
                static_cast< float_X >( 2 * T_numNodes ) *
                picongpu::PI
                );
        }


        //return unitless
        DINLINE static float_64 fak ( const uint32_t n )
        {
            float_64 result = 1u;

            for ( uint32_t i = 1u; i <= n; i++ )
            {
                result *= static_cast< float_64 >( i );
            }

            return result;
        }

    public:

        // calculates weightings once
        DINLINE void init( )
        {
            float_X samplePoints[ T_numSamplePoints ];

            // include reference point
            samplePoints[ 0 ] = 0._X;

            for ( uint32_t i = 1u; i < T_numSamplePoints; i++ )
            {
                samplePoints[ i ] = chebyshevNodes< ( T_numSamplePoints - 1u ) >( i );
            }

            // calculate weightings for each order derivative until maximum
            for ( uint32_t i = 0u; i <= ( 2u * T_maxOrderApprox ); i++ )    // i ... order of derivation
            {
                for ( uint32_t j = 0u; j < T_numSamplePoints; j++ )  // j ... sample point
                {
                    this->weights[ j + i * T_numSamplePoints ] = T_WeightingGen::template weighting<
                        T_numSamplePoints,
                        T_numSamplePoints/2u + 2u >( i, j, samplePoints );
                }
            }
        }


        // return unit: [1/s]/[ 1/(m^3 * J) ], error per electron density
        template< typename T_Acc >
        DINLINE float_X operator() (
            T_Acc & acc,
            float_X dE,     // unit: ATOMIC_UNIT_ENERGY
            float_X E,       // unit: ATOMIC_UNIT_ENERGY
            T_AtomicDataBox atomicDataBox
            ) const
        {
            float_X result = 0._X; // unit: (1/s) / (m^3/J)

            // a ... order of rate approximation
            for ( uint32_t a = T_minOrderApprox; a <= T_maxOrderApprox; a++ )
            {
                // o ... derivative order of crossection sigma
                for ( uint32_t o = 0u; o <= 2 * a; o++ )
                    {
                        // taylor expansion of rate integral, see my master thesis
                        // \sum{ 1/(o! (2a-o)!) * sigma'(o) * v'(2a-o) * 1/(2a+1) * (dE/2)^(2a+1) * 2}
                        // 1/unitless * m^2/J^o * m/s * 1/J^(2*a-o) * unitless * J^(2*a+1) * unitless
                        // = m^3/J^(2a) * 1/s * J^(2a+1) = J * m^3 * 1/s
                        result += 1.0/( fak( o ) * fak( 2u*a - o ) ) *
                            sigmaDerivative(acc, E, dE, o, atomicDataBox) *
                            velocityDerivative( E, dE, 2u*a - o ) *
                            1._X/( 2u*a + 1u) *
                            mathFunc::pow(
                                dE * picongpu::SI::ATOMIC_UNIT_ENERGY / 2._X,
                                static_cast< int >( 2u*a + 1u)
                                ) * 2._X; // unit: J * m^3 * 1/s
                    }
            }
            //printf("    deltaE: %d\n", dE);
            //printf("    relative error: %d\n", result);
            return result;
        }

    private:

        // @param energy ... unit: atomic energy units
        // return unit: m/s, SI
        DINLINE static float_X velocity(
            float_X energy  // unit: atomic energy unit
            )
        {
            constexpr float_X mass_e_SI = picongpu::SI::ELECTRON_MASS_SI;
            constexpr float_X c_SI = picongpu::SI::SPEED_OF_LIGHT_SI;

            float_X restEnergy_SI = mass_e_SI * pmacc::algorithms::math::pow( c_SI, 2 );

            // v = sqrt( 1 - (m^2*c^4)/(E^2) )
            return picongpu::SI::SPEED_OF_LIGHT_SI *
                mathFunc::sqrt( 1._X - mathFunc::pow(
                    1._X / ( 1._X + (energy * picongpu::SI::ATOMIC_UNIT_ENERGY) /
                    restEnergy_SI ),
                    2
                    )
                    );
        }

        //return unit: m/s * 1/J^m), SI
        DINLINE float_X velocityDerivative(
            float_X E,      // unit: ATOMIC_UNIT_ENERGY
            float_X dE,     // unit: ATOMIC_UNIT_ENERGY
            uint32_t m     // order of derivative
            ) const
        {
            // samplePoint[ 0 ], is by definition always = 0:
            float_X weight = this->weights[ m * T_numSamplePoints ]; // unit: unitless

            float_X velocityValue = velocity( E ); // unit: m/s, SI

            float_X result = weight * velocityValue; // unit: m/s, SI

            // all further sample points
            for( uint32_t j = 1u; j < T_numSamplePoints; j++ )  // j index of sample point
            {
                weight = this->weights[ m * T_numSamplePoints + j ]; // unit: unitless

                // velocity( [ ATOMIC_UNIT_ENERGY ] ) -> m/s, SI
                float_X velocityValue = velocity(
                    E + chebyshevNodes< T_numSamplePoints - 1u >( j + 1u ) * dE / 2._X
                    ); // unit: m/s, SI

                result += weight * velocityValue;   // unit: m/s, SI
            }

            // pow( 1/[ ATOMIC_UNIT_ENERGY * AU_To_J ] = 1/J, m )-> 1/(J^m)
            result /= mathFunc::pow(
                dE / 2._X * picongpu::SI::ATOMIC_UNIT_ENERGY,
                int( m ) ); // unit: m/s * 1/J^m, SI

            return result;
        }


        // return unit: m^2/J^m, SI
        template< typename T_Acc >
        DINLINE float_X sigmaDerivative(
            T_Acc & acc,
            float_X E,      // central energy of bin, unit: ATOMIC_UNIT_ENERGY
            float_X dE,     // delta energy, unit: ATOMIC_UNIT_ENERGY
            uint32_t o,     // order of derivative, unitless
            T_AtomicDataBox const atomicDataBox //in file atomicData.hpp
           ) const
        {

            // samplePoint[ 0 ], is by definition always = 0
            float_X weight = this->weights[ o * T_numSamplePoints ]; // unit: unitless

            //float_X sigmaValue = AtomicRate::totalCrossSection(0u);
            float_X sigmaValue = AtomicRate::totalCrossSection(
                acc,
                E,  // unit: ATOMIC_UNIT_ENERGY
                atomicDataBox
                ); // unit: m^2, SI

            float_X result = weight * sigmaValue; // unit: m^2, SI
            const auto numSamplePoints = T_numSamplePoints;

            // all further sample points
            for( uint32_t j = 1u; j < T_numSamplePoints; j++ )  // j index of sample point
            {
                weight = this->weights[ o * numSamplePoints + j ]; // unitless

                //sigmaValue = AtomicRate::totalCrossSection< T_ConfigNumberDataType >();
               //         E + chebyshevNodes< T_numSamplePoints - 1u >( j + 1u ) * dE / 2._X,
                //        atomicDataBox
               //         ); // unit: m^2, SI

                result += weight * sigmaValue; // unit: m^2, SI
            }
            // m^2 / (J)^m
            result /= mathFunc::pow( dE * picongpu::SI::ATOMIC_UNIT_ENERGY / 2._X, int(o) );

            return result; // unit: m^2/J^m, SI
        }
    };

} // namespace histogram2
} // namespace electronDistribution
} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu

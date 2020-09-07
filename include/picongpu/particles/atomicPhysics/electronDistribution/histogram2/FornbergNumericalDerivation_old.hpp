/* Copyright 2019-2020 Brian Marre
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

/** @file */

#pragma once

#include <pmacc/algorithms/math.hpp>
#include <utility>

namespace picongpu
{
namespace partciles
{
namespace atomicPhysics
{
namespace electronDistribution
{
namespace histogram2
{

    // encapsulates the numerical derivation of a callable function
    template<
        uint32_t T_numSamplePoints,
        uint32_t T_orderDerivative,
        typename T_Value,
        typename T_Argument
        >
    struct FornbergNumericalDifferentation
    {
    private:
        // make template parameters available for further use
        constexpr static uint32_t numSamplePoints = T_numSamplepoints;
        constexpr static uint32_t orderDerivative = T_orderDerivative;

        // sample points s_i are defined as:
        // the x_i = x_ref + s_i * d
        T_Argument samplePoints [ numSamplePoints ];
        T_Value coefficients[ numSamplePoints ];

    public:
        // constructor for user given sample points
        FornbergNumericalDifferentation(
            T_Argument & const samplePoints[ numSamplePoints ]
            )
        {
            for ( int i=0; i < numSamplePoints; i++ )
            {
                // store user given sample points
                this->samplePoints[i] = samplePoints[i];
                // TODO: implement coefficient calculation
                this->coefficients[i] = static_cast< T_Value >(0);
            }
        }

//        // constructor using chebyshev nodes as sampling points
//        // requires the function to be callable at all chebyshev points
//        FornbergNumericalDifferentation(
//            T_Argument centralValue,
//            T_Argument sampleIntervalWidth
//            )
//        {
//            for ( uint32_t i = 0; i < numSamplePoints; i++ )
//            {
//                this->samplePoints[i] = scaledChebyshevNodes< numSamplePoints >(
//                    i,
//                    centralValue,
//                    sampleIntervalWidth
//                    )
//            }

            // TODO: make this a wrapper of class constructor
            // call direct constructor here instead of copy paste
            // TODO: implement coefficient calculation
        }

//        // returns the chebyshev node postions
//        // if k out of bounds returns nearest interval boundary
//        template<
//            uint32_t T_numNodes
//        >
//        DINLINE static float_X scaledChebyshevNodes(
//            uint32_t k,
//            float_X centralX,
//            float_X deltaX
//        ) const
//        {
//            /** returns the k-th of T_numNodes scaled chebyshev nodes x_k of the interval
//            * [centralX - deltaX/2, centralX + deltaX/2] = [a, b]
//            *
//            * x_k= 1/2 *(a+b) + 1/2 * (b-a) * cos( (2k-1)/(2n) * pi )
//            *
//            * @param k ... index of chebyshev Nodes
//            * @param centralX ... central value of interval, = 1/2*(a+b) = a + 1/2*(b-a)
//            * @param deltaX ... width of intervall, = b-a
//            *
//            * @tparam T_numNodes ... how many chebyshev nodes are requested
//            *
//            * BEWARE: k=0 is NOT allowed, k = 1, ... , T_numNodes
//            * BEWARE: the node with the highest argument value in the interval corresponds
//            *   to the lowest k value
//            *
//            * see https://en.wikipedia.org/wiki/Chebyshev_nodes for more information
//            */
//
//            // check for bounds on k
//            // return nearest boundaries if outside of boundary
//            if ( k < 1 )
//                return centralX + deltaX/2._X;
//            if ( k > N )
//                return centralX - deltaX/2._X;
//
//            return pmacc::algorithms::math::cos< float_X >(
//                (2*k - 1)_X/( 2 * T_numNodes )_X * pmacc::algorithms::math::Pi::value ) *
//                deltaX / 2._X + centralX;
//        }

        T_Argument getSamplePoint( uint32_t index){
            if ( index )
            return this->samplePoints[ index ];
        }

        template<
            typename T_Function
            >
        static T_Value getDerivative(
            T_Function const & function,
            T_Argument argument,
            uint8_t orderDerivative,
            )
        {
            /**returns derivative of a callable function at the argument
             *
             * @tparam T_Function ... type of function
             *
             * @param function ... callable function,
             *      must define: T_Value operator()( T_Argument x )
             *      returning the value of function for x, must at least cover all 
             *      sample points.
             * @param argument ... point where to calculate the derivative
             * @param sampleIntervalWidth ... width of the intervall centered 
             * @param orderDerivative ... order of derivative to be calculated
             */

            T_Value result = static_cast< T_Value >( 0 );

            for ( uint8_t samplePoint = 0; samplePoint <= T_numSamples; samplePoint++ )
            {
                result += this->coefficients[ samplePoint ] * function( samplePoints[i] );
            }

            return result;
        }

        template<
            uint8_t T_orderDerivative,
            uint8_t T_orderDerivative
            >
        static float_X numericalDerivativeCoefficients()
        {
            return 1._X
        }
    }

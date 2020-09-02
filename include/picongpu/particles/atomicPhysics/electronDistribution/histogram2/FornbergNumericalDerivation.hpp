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
    struct FornbergNumericalDifferentation
    {

        // returns the chebyshev node postions
        // if k out of bounds returns nearest interval boundary
        template<
            uint8_t T_numNodes
        >
        DINLINE static float_X scaledChebyshevNodes(
            uint8_t k,
            float_X centralX,
            float_X deltaX
        ) const
        {
            /** returns the k-th of T_numNodes scaled chebyshev nodes x_k of the interval
            * [centralX - deltaX/2, centralX + deltaX/2] = [a, b]
            *
            * x_k= 1/2 *(a+b) + 1/2 * (b-a) * cos( (2k-1)/(2n) * pi )
            *
            * @param k ... index of chebyshev Nodes
            * @param centralX ... central value of interval, = 1/2*(a+b) = a + 1/2*(b-a)
            * @param deltaX ... width of intervall, = b-a
            *
            * @tparam T_numNodes ... how many chebyshev nodes are requested
            *
            * BEWARE: k=0 is NOT allowed, k = 1, ... , T_numNodes
            * BEWARE: the node with the highest argument value in the interval corresponds
            *   to the lowest k value
            *
            * see https://en.wikipedia.org/wiki/Chebyshev_nodes for more information
            */
    
            // check for bounds on k
            // return nearest boundaries if outside of boundary
            if ( k < 1 )
                return centralX + deltaX/2._X;
            if ( k > N )
                return centralX - deltaX/2._X;

            return pmacc::algorithms::math::cos< float_X >(
                (2*k - 1)_X/( 2 * T_numNodes )_X * pmacc::algorithms::math::Pi::value ) *
                deltaX / 2._X + centralX;
        }

        template<
            typename T_Function,
            typename T_Argument,
            uint8_t T_orderDerivative,
            uint8_t T_numSamples
            >
        static float_X getDerivative(
            T_Function & const function,
            T_Argument argument
            )
        {
            return 0._X;
        }

        static float_X numericalDerivativeCoefficients()
        {
            return 1._X
        }
    }

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

/** @file This file defines a Wrapper for numerical differentiation weighting
 * generators.
 *
 * primarily this wrapper stores the sample points togehter with the actual weighting
 * generator.
 * use generator directly if you already have acess to sample points
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
                    // @tparam T_WeightingGen ... has method T_Value weighting( index, samplePoints[ T_numSamplePoints
                    // ] )
                    template<
                        typename T_WeightingGen, // partial specialisation leaving order as remaining parameter
                        uint32_t T_maxOrderDerivative, // must be <= T_numSamplePoints - 1
                        uint32_t T_numSamplePoints,
                        typename T_Value,
                        typename T_Argument>
                    class numericalDerivative
                    {
                        // realtive sample points s_i as multiples of scaling
                        T_Argument samplePoints[T_numSamplePoints];

                        // T_Value weights [ T_numSamplePoints ]

                    public:
                        // constructor for user given sample points
                        DINLINE constexpr numericalDerivative(T_Argument const samplePoints[T_numSamplePoints]) const
                        {
                            for(int i = 0; i < T_numSamplePoints; i++)
                            {
                                // store user sample points
                                this->samplePoints[i] = samplePoints[i];
                            }
                        }

                        // constructor for chebyshev sample points, use chebyshevNodes( i )
                        DINLINE constexpr numericalDerivative() const
                        {
                            for(uint32_t i = 0; i < T_numSamplePoints; i++)
                            {
                                // store user sample points
                                this->samplePoints[i] = scaledChebyshevNodes(i + 1u);
                            }
                        }

                        // Acessors
                        // get relative Sample Points
                        DINLINE constexpr T_Argument getRelativeSamplePoints(uint32_t const index)
                        {
                            if(index < T_numSamplePoints)
                                return this->samplePoints[index];

                            return this->samplePoints[T_numSamplePoints - 1u];
                        }

                        // get Weighting of sample Point
                        DINLINE constexpr T_Value getWeighting(uint32_t const index, uint32_t orderDerivative)
                        {
                            return T_WeightingGen<orderDerivative>::weighting(index, this->samplePoints);
                        }

                        /** returns derivative uing array  of function values at samplePoints
                         *
                         * @param functionValues value of function to
                         *      differentiate at the sample points
                         * @param orderDerivative .. order of derivative to caluculate
                         */
                        DINLINE constexpr T_Value derivativeArray(
                            T_Value const functionValues[T_numSamplePoints],
                            uint32_t orderDerivative)
                        {
                            T_Value result = static_cast<T_Value>(0);

                            for(uint32_t i = 0u; i < T_numSamples; i++)
                            {
                                result += T_WeightingGen<orderDerivative>::weighting(i, this->samplePoints)
                                    * functionValue[i];
                            }

                            return result;
                        }

                        /** returns numerical value of the derivative of a given order of a
                         * functor for the argument
                         *
                         * @tparam T_Function ... Functor of function
                         *
                         * @param argument ... argument at which to differentiate
                         * @param orderDerivative ... order of derivative to be calculated
                         */
                        template<typename T_Function>
                        DINLINE constexpr T_Value derivativeFunctor(T_Argument argument, uint32_t orderDerivative)
                        {
                            T_Value result = static_cast<T_Value>(0);

                            for(uint32_t i = 0u; i < T_numSamples; i++)
                            {
                result += T_WeightingGen::weighting< // TODO: add >(
                    i,
                    this->samplePoints
                    )
                    * T_Function( this->samplePoints[ i ] * scaling + argument );
                            }

                            return result;
                        }

                        // returns chebyshev node positions
                        // if k out of bounds returns nearest interval boundary
                        template<uint32_t T_numNodes>
                        DINLINE static constexpr T_Value chebyshevNodes(uint32_t k) const
                        {
                            /** returns the k-th of T_numNodes chebyshev nodes x_k, interval [-1, 1]
                             *
                             * x_k = cos( (2k-1)/(2n) * pi )
                             *
                             * @tparam T_numNodes ... how many chebyshev nodes are requested
                             * @param k ... index of chebyshev Nodes
                             *
                             * BEWARE: k=0 is NOT allowed, k \in {1, ... , T_numNodes}
                             * BEWARE: max({ x_k }) = x_1 and min({ x_k }) = x_T_numNodes
                             *
                             * see https://en.wikipedia.org/wiki/Chebyshev_nodes for more information
                             */

                            // check for bounds on k
                            // return boundaries if outside of boundary
                            if(k < 1)
                                return 0.;
                            if(k > N)
                                return 1.;

                            return static_cast<T_Value>(math::cos<float_X>(
                                (2 * k - 1) _X / (2 * T_numNodes) _X * math::Pi::value));
                        }
                    };

                } // namespace histogram2
            } // namespace electronDistribution
        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu

/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Alexander Grund, Sergei Bastrakov
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


namespace picongpu
{
    namespace particles
    {
        namespace manipulators
        {
            namespace unary
            {
                namespace acc
                {
                    /** Modify particle momentum based on temperature
                     *
                     * Generate a new random momentum distributed according to the given
                     * temperature and add it to the existing particle momentum.
                     * This functor is for the non-relativistic case only.
                     * In this case the new momentums follow the Maxwell-Boltzmann distribution.
                     *
                     * @tparam T_ParamClass picongpu::particles::manipulators::unary::param::TemperatureCfg,
                     *                      type with compile configuration
                     * @tparam T_ValueFunctor pmacc::nvidia::functors::*, binary functor type to
                     *                        add a new momentum to an old one
                     */
                    template<typename T_ParamClass, typename T_ValueFunctor>
                    struct Temperature : private T_ValueFunctor
                    {
                        /** manipulate the speed of the particle
                         *
                         * @tparam T_StandardNormalRng functor::misc::RngWrapper, standard
                         *                             normal random number generator type
                         * @tparam T_Particle pmacc::Particle, particle type
                         * @tparam T_Args pmacc::Particle, arbitrary number of particles types
                         *
                         * @param rng standard normal random number generator
                         * @param particle particle to be manipulated
                         * @param ... unused parameters
                         */
                        template<typename T_StandardNormalRng, typename T_Particle, typename... T_Args>
                        HDINLINE void operator()(
                            T_StandardNormalRng& standardNormalRng,
                            T_Particle& particle,
                            T_Args&&...)
                        {
                            /* In the non-relativistic case, particle momentums are following
                             * the Maxwell-Boltzmann distribution: each component is
                             * independently normally distributed with zero mean and variance of
                             * m * k * T = m * E.
                             * For the macroweighted momentums we store as particle[ momentum_ ],
                             * the same relation holds, just m and E are also macroweighted
                             */
                            float_X const energy = (T_ParamClass::temperature * UNITCONV_keV_to_Joule) / UNIT_ENERGY;
                            float_X const macroWeighting = particle[weighting_];
                            float_X const macroEnergy = macroWeighting * energy;
                            float_X const macroMass = attribute::getMass(macroWeighting, particle);
                            float_X const standardDeviation
                                = static_cast<float_X>(math::sqrt(precisionCast<sqrt_X>(macroEnergy * macroMass)));
                            float3_X const mom
                                = float3_X(standardNormalRng(), standardNormalRng(), standardNormalRng())
                                * standardDeviation;
                            T_ValueFunctor::operator()(particle[momentum_], mom);
                        }
                    };

                } // namespace acc
            } // namespace unary
        } // namespace manipulators
    } // namespace particles
} // namespace picongpu

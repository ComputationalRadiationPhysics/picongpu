/* Copyright 2013-2021 Axel Huebl, Rene Widera
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
                    /** manipulate the speed
                     *
                     * @tparam T_ParamClass picongpu::particles::manipulators::unary::param::DriftCfg,
                     *                      type with compile configuration
                     * @tparam T_ValueFunctor pmacc::nvidia::functors, binary operator type to reduce current and new
                     * value
                     */
                    template<typename T_ParamClass, typename T_ValueFunctor>
                    struct Drift : private T_ValueFunctor
                    {
                        /** manipulate the speed of the particle
                         *
                         * @tparam T_Particle pmacc::Particle, particle type
                         * @tparam T_Args pmacc::Particle, arbitrary number of particles types
                         *
                         * @param particle particle to be manipulated
                         * @param ... unused particles
                         */
                        template<typename T_Particle, typename... T_Args>
                        HDINLINE void operator()(T_Particle& particle, T_Args&&...)
                        {
                            using ParamClass = T_ParamClass;
                            using ValueFunctor = T_ValueFunctor;

                            float_X const macroWeighting = particle[weighting_];
                            float_X const macroMass = attribute::getMass(macroWeighting, particle);

                            float_64 const myGamma = ParamClass::gamma;

                            float_64 const initFreeBeta = math::sqrt(1.0 - 1.0 / (myGamma * myGamma));

                            float3_X const driftDirection(ParamClass().direction);
                            float3_X const normDir = driftDirection / math::abs(driftDirection);

                            float3_X const mom(
                                normDir
                                * float_X(myGamma * initFreeBeta * float_64(macroMass) * float_64(SPEED_OF_LIGHT)));

                            ValueFunctor::operator()(particle[momentum_], mom);
                        }
                    };

                } // namespace acc
            } // namespace unary
        } // namespace manipulators
    } // namespace particles
} // namespace picongpu

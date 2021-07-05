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

#include "picongpu/param/physicalConstants.param"
#include "picongpu/particles/manipulators/unary/PhotonDrift.def"

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
                    /** manipulate momentum
                     *
                     * @tparam T_ParamClass   type with compile configuration
                     * @tparam T_ValueFunctor  binary operator type to reduce current and new value
                     */
                    template<typename T_ParamClass, typename T_ValueFunctor>
                    struct PhotonDrift : private T_ValueFunctor
                    {
                        /** manipulate momentum
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
                            float_64 const photonMomentum{
                                ParamClass().photonEnergySI / UNIT_ENERGY / static_cast<float_64>(SPEED_OF_LIGHT)};
                            float3_X const driftDirection{ParamClass().direction};
                            float3_X const normDir{driftDirection / math::abs(driftDirection)};
                            float3_X const mom{normDir * (static_cast<float_X>(photonMomentum) * macroWeighting)};

                            ValueFunctor::operator()(particle[momentum_], mom);
                        }
                    };

                } // namespace acc
            } // namespace unary
        } // namespace manipulators
    } // namespace particles
} // namespace picongpu

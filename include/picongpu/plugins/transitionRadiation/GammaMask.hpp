/* Copyright 2017-2021 Rene Widera, Finn-Ole Carstens
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

#include <pmacc/traits/HasIdentifier.hpp>


namespace picongpu
{
    namespace plugins
    {
        namespace transitionRadiation
        {
            /** read the `transitionRadiationMask` of a species */
            template<bool hasTransitionRadiationMask>
            struct GetTransitionRadiationMask
            {
                /** get the attribute value of `transitionRadiationMask`
                 *
                 * @param particle particle to be used
                 * @return value of the attribute `transitionRadiationMask`
                 */
                template<typename T_Particle>
                HDINLINE bool operator()(const T_Particle& particle) const
                {
                    return particle[transitionRadiationMask_];
                }
            };

            /** specialization
             *
             * specialization for the case that the species not owns the attribute
             * `transitionRadiationMask`
             */
            template<>
            struct GetTransitionRadiationMask<false>
            {
                /** get the attribute value of `transitionRadiationMask`
                 *
                 * @param particle to be used
                 * @return always true
                 */
                template<typename T_Particle>
                HDINLINE bool operator()(const T_Particle&) const
                {
                    return true;
                }
            };

            /** get the value of the particle attribute `transitionRadiationMask`
             *
             * Allow to read out the value of the attribute `transitionRadiationMask` also if
             * it is not defined for the particle.
             *
             * @tparam T_Particle particle type
             * @param particle valid particle
             * @return particle attribute value `transitionRadiationMask`, always `true` if attribute
             * `transitionRadiationMask` is not defined
             */
            template<typename T_Particle>
            HDINLINE bool getTransitionRadiationMask(const T_Particle& particle)
            {
                constexpr bool hasTransitionRadiationMask = pmacc::traits::
                    HasIdentifier<typename T_Particle::FrameType, transitionRadiationMask>::type::value;
                return GetTransitionRadiationMask<hasTransitionRadiationMask>{}(particle);
            }

        } // namespace transitionRadiation
    } // namespace plugins
} // namespace picongpu

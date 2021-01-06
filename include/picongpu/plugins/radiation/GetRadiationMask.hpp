/* Copyright 2017-2021 Rene Widera
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
        namespace radiation
        {
            /** read the `radiationMask` of a species */
            template<bool hasRadiationMask>
            struct GetRadiationMask
            {
                /** get the attribute value of `radiationMask`
                 *
                 * @param particle particle to be used
                 * @return value of the attribute `radiationMask`
                 */
                template<typename T_Particle>
                HDINLINE bool operator()(const T_Particle& particle) const
                {
                    return particle[picongpu::radiationMask_];
                }
            };

            /** specialization
             *
             * specialization for the case that the species not owns the attribute
             * `radiationMask`
             */
            template<>
            struct GetRadiationMask<false>
            {
                /** get the attribute value of `radiationMask`
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

            /** get the value of the particle attribute `radiationMask`
             *
             * Allow to read out the value of the attribute `radiationMask` also if
             * it is not defined for the particle.
             *
             * @tparam T_Particle particle type
             * @param particle valid particle
             * @return particle attribute value `radiationMask`, always `true` if attribute `radiationMask` is not
             * defined
             */
            template<typename T_Particle>
            HDINLINE bool getRadiationMask(const T_Particle& particle)
            {
                constexpr bool hasRadiationMask
                    = pmacc::traits::HasIdentifier<typename T_Particle::FrameType, radiationMask>::type::value;
                return GetRadiationMask<hasRadiationMask>{}(particle);
            }
        } // namespace radiation
    } // namespace plugins
} // namespace picongpu

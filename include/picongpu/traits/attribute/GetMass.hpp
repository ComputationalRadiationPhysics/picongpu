/* Copyright 2014-2021 Rene Widera
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
#include "picongpu/traits/frame/GetMass.hpp"


namespace picongpu
{
    namespace traits
    {
        namespace attribute
        {
            /** get the mass of a makro particle
             *
             * @param weighting weighting of the particle
             * @param particle a reference to a particle
             * @return mass of the makro particle
             */
            template<typename T_Particle>
            HDINLINE float_X getMass(const float_X weighting, const T_Particle& particle)
            {
                using ParticleType = T_Particle;
                return frame::getMass<typename ParticleType::FrameType>() * weighting;
            }

        } // namespace attribute
    } // namespace traits
} // namespace picongpu

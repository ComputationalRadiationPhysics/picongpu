/* Copyright 2021-2022 Brian Marre
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "picongpu/simulation_defines.hpp"

#include "picongpu/traits/attribute/GetMass.hpp"

#pragma once

namespace picongpu::particles::atomicPhysics
{
    struct GetPhysicalEnergy
    {
        /** returns the relativistic kinetic energy of a corresponding single physical particle
         *
         * @tparam T_Particle type of given macroParticle
         *
         * @param particle macroParticle corresponding to the physical particle
         *
         * @attention is not optimised for highly relativistic particles, expect imprecise/wrong
         *   results for energies much larger than GeV
         *
         * @return [eV]
         */
        template<typename T_Particle>
        HDINLINE static float_X KineticEnergy(T_Particle const& particle)
        {
            // is by definition == 1 in current internal unit system
            /// @todo convert to assert/ compile time check with special case, Brian Marre, 2023
            constexpr float_X conversionFactor = static_cast<float_X>(
                (picongpu::sim.unit.length() * picongpu::sim.unit.length())
                / (picongpu::sim.unit.time() * picongpu::sim.unit.time() * picongpu::SI::SPEED_OF_LIGHT_SI
                   * picongpu::SI::SPEED_OF_LIGHT_SI));

            // UNIT_MASS, not scaled
            float_X const m = picongpu::traits::frame::getMass<typename T_Particle::FrameType>();

            // UNIT_MASS * sim.unit.length() / sim.unit.time(), scaled
            float3_X vectorMomentum_Scaled = particle[momentum_];

            // UNIT_MASS^2 * sim.unit.length()^2 / sim.unit.time()^2, not scaled
            float_X momentumSquared
                = pmacc::math::l2norm2(vectorMomentum_Scaled) / (particle[weighting_] * particle[weighting_]);

            // float_X should be sufficient,
            // m^2 + p^2 = (<=80 * ~2000)^2 + (<2000[10^9eV])^2 ~ 2.5*10^10
            //{
            // sim.unit.length()^2 / (sim.unit.time()^2*c_SI) = 1
            // UNIT_MASS * c_SI^2
            float_X const energy = (math::sqrt(m * m + momentumSquared * conversionFactor) - m);
            //}

            // unit conversion factor for eV
            constexpr float_X eV = static_cast<float_X>(
                picongpu::UNIT_MASS * picongpu::SI::SPEED_OF_LIGHT_SI * picongpu::SI::SPEED_OF_LIGHT_SI
                * picongpu::UNITCONV_Joule_to_keV * 1e3);

            // eV
            return energy * eV;
        }
    };
} // namespace picongpu::particles::atomicPhysics

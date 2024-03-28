/* Copyright 2021-2023 Brian Marre
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

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            struct GetRealKineticEnergy
            {
                // returns the relativistic kinetic energy of a single physical particle
                // represented by the given macro particle
                // return unit: J, SI
                template<typename T_Particle>
                HDINLINE static float_64 KineticEnergy(T_Particle const& particle)
                {
                    constexpr float_64 c_SI = picongpu::SI::SPEED_OF_LIGHT_SI; // unit: m/s, SI

                    // mass of physical particle
                    float_64 m_p_SI_rel = attribute::getMass(1.0_X, particle) * picongpu::UNIT_MASS * c_SI * c_SI;
                    // unit: J, SI

                    float3_X vectorMomentum_Scaled = particle[momentum_]; // unit: internal, scaled
                    float_X momentumSquared
                        = pmacc::math::l2norm2(vectorMomentum_Scaled) / (particle[weighting_] * particle[weighting_]);
                    // unit: internal, not scaled

                    float_64 momentumSquared_p_SI_rel = c_SI * c_SI
                        * (momentumSquared * picongpu::UNIT_MASS * picongpu::UNIT_MASS * picongpu::UNIT_LENGTH
                           * picongpu::UNIT_LENGTH
                           / (picongpu::UNIT_TIME * picongpu::UNIT_TIME)); // unit: J^2, SI, not scaled

                    // TODO: switch to ATOMIC_UNIT_ENERGY for intermediate calculations, Brian Marre, 2021

                    // relativistic kinetic energy
                    // E_kin = sqrt((m*c^2)^2 + (p*c)^2) - m*c^2
                    return math::sqrt(m_p_SI_rel * m_p_SI_rel + momentumSquared_p_SI_rel) - m_p_SI_rel;
                    // sqrt( (kg * m^2/s^2)^2  + (kg * m/s * m/s)^2 ) - kg*m^2/s^2
                    // unit: kg * m^2/s^2 = J, SI
                }
            };

        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu

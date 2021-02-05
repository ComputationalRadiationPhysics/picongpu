/* Copyright 2020 Brian Marre
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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"

#pragma once

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            namespace mathFunc = pmacc::algorithms::math;

            struct GetRealKineticEnergy
            {
                // returns the kinetic energy of a represented physical particle
                // return unit: J, SI
                template<typename T_Particle>
                HDINLINE static float_X KineticEnergy(T_Particle& particle)
                {
                    constexpr auto c_SI = picongpu::SI::SPEED_OF_LIGHT_SI; // unit: m/s, SI

                    // unit: kg, SI
                    auto m_p_SI = attribute::getMass(1.0_X, particle) * picongpu::SI::BASE_MASS_SI;


                    float3_X vectorMomentum_Scaled = particle[momentum_]; // internal units and scaled with weighting

                    float_X momentum = math::abs2(vectorMomentum_Scaled) / particle[weighting_]; // internal units

                    // TODO: check wheter conversion is correct
                    // unit: kg * m/s, SI
                    float_X momentum_SI = momentum * picongpu::UNIT_MASS * picongpu::UNIT_LENGTH / picongpu::UNIT_TIME;

                    // TODO: note about math functions:
                    // in the dev branch need to add pmacc:: and acc as first parameter

                    // sqrt( kg^2 * m^4/s^4  + (kg * m/s * m/s)^2 )
                    return math::sqrt(
                               mathFunc::pow(m_p_SI, 2) * mathFunc::pow(c_SI, 4)
                               + mathFunc::pow(momentum_SI * c_SI, 2))
                        - m_p_SI * mathFunc::pow(c_SI, 2); // - m*c^2, since kinetic energy is what we want
                    // unit: kg * m^2/s^2 = J, SI
                }
            };

        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu

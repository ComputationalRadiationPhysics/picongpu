/* Copyright 2013-2023 Axel Huebl, Rene Widera, Richard Pausch
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

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/IsWeighted.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/LarmorPower.def"

#include <pmacc/static_assert.hpp>

#include <type_traits>


namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace derivedAttributes
            {
                HDINLINE float1_64 LarmorPower::getUnit() const
                {
                    return sim.unit.energy();
                }

                template<class T_Particle>
                DINLINE float_X LarmorPower::operator()(T_Particle& particle) const
                {
                    constexpr bool hasMomentumPrev1
                        = pmacc::traits::HasIdentifier<typename T_Particle::FrameType, momentumPrev1>::type::value;
                    PMACC_CASSERT_MSG_TYPE(
                        species_must_have_the_attribute_momentumPrev1,
                        T_Particle,
                        hasMomentumPrev1);

                    /* read existing attributes */
                    const float3_X mom = particle[momentum_];
                    const float3_X mom_mt1 = particle[momentumPrev1_];
                    const float_X weighting = particle[weighting_];
                    const float_X charge = picongpu::traits::attribute::getCharge(weighting, particle);
                    const float_X mass = picongpu::traits::attribute::getMass(weighting, particle);

                    /* calculate new attribute */
                    Gamma<float_X> calcGamma;
                    const typename Gamma<float_X>::valueType gamma = calcGamma(mom, mass);
                    const float_X gamma2 = gamma * gamma;
                    const float_X c2 = sim.pic.getSpeedOfLight() * sim.pic.getSpeedOfLight();

                    const float3_X mom_dt = (mom - mom_mt1) / float_X(sim.pic.getDt());
                    const float_X el_factor = charge * charge
                        / (float_X(6.0) * PI * sim.pic.getEps0() * c2 * sim.pic.getSpeedOfLight() * mass * mass)
                        * gamma2 * gamma2;
                    const float_X momentumToBetaConvert = float_X(1.0) / (mass * sim.pic.getSpeedOfLight() * gamma);
                    const float_X larmorPower = el_factor
                        * (pmacc::math::l2norm2(mom_dt)
                           - momentumToBetaConvert * momentumToBetaConvert
                               * pmacc::math::l2norm2(pmacc::math::cross(mom, mom_dt)));

                    /* return attribute */
                    return larmorPower;
                }

                //! Larmor power is weighted
                template<>
                struct IsWeighted<LarmorPower> : std::true_type
                {
                };
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu

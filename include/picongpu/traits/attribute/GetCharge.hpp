/* Copyright 2014-2023 Rene Widera, Axel Huebl
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/traits/GetAtomicNumbers.hpp"
#include "picongpu/traits/frame/GetCharge.hpp"

#include <pmacc/static_assert.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/HasIdentifier.hpp>


namespace picongpu
{
    namespace traits
    {
        namespace attribute
        {
            /** get the charge of a macro particle
             *
             * This function trait considers the `boundElectrons` attribute if it is set
             *
             * @param weighting weighting of the particle
             * @param particle a reference to a particle
             * @return charge of the macro particle
             */
            template<typename T_Particle>
            HDINLINE float_X getCharge(const float_X weighting, const T_Particle& particle)
            {
                using ParticleType = T_Particle;
                constexpr bool hasBoundElectrons
                    = pmacc::traits::HasIdentifier<ParticleType, boundElectrons>::type::value;
                if constexpr(hasBoundElectrons)
                {
                    using HasAtomicNumbers = typename pmacc::traits::HasFlag<T_Particle, atomicNumbers<>>::type;
                    PMACC_CASSERT_MSG_TYPE(
                        Having_boundElectrons_particle_attribute_requires_atomicNumbers_flag,
                        T_Particle,
                        HasAtomicNumbers::value);
                    const float_X protonNumber = GetAtomicNumbers<T_Particle>::type::numberOfProtons;

                    /* note: sim.pic.getElectronCharge() is negative and the second term is also negative
                     */
                    return sim.pic.getElectronCharge() * (particle[boundElectrons_] - protonNumber) * weighting;
                }

                return frame::getCharge<typename T_Particle::FrameType>() * weighting;
            }

        } // namespace attribute
    } // namespace traits
} // namespace picongpu

/*
 * SPDX-FileCopyrightText: Rene Widera, Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
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
            HDINLINE float_X getCharge(float_X const weighting, T_Particle const& particle)
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
                    float_X const protonNumber = picongpu::traits::GetAtomicNumbers<T_Particle>::type::numberOfProtons;

                    /* note: sim.pic.getElectronCharge() is negative and the second term is also negative
                     */
                    return sim.pic.getElectronCharge() * (particle[boundElectrons_] - protonNumber) * weighting;
                }

                return traits::frame::getCharge<typename T_Particle::FrameType>() * weighting;
            }

        } // namespace attribute
    } // namespace traits
} // namespace picongpu

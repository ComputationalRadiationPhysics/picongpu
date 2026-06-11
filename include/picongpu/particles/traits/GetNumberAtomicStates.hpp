/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/particles/atomicPhysics/ParticleType.hpp"

#include <pmacc/particles/memory/frames/Frame.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <cstdint>

namespace picongpu::traits
{
    /** @class compile time functor for accessing data in numberAtomicStates flag of species
     *
     * @tparam T_IonSpecies resolved typename of species with flag
     * @returns return value contained in ::value
     */
    template<typename T_IonSpecies>
    struct GetNumberAtomicStates
    {
        using FrameType = typename T_IonSpecies::FrameType;

        /* throw static assert if species lacks flag */
        PMACC_CASSERT_MSG(
            This_species_is_not_marked_as_an_atomicPhysics_ion_species,
            particles::atomicPhysics::traits::IsParticleType<
                particles::atomicPhysics::traits::GetParticleType_t<FrameType>,
                particles::atomicPhysics::Tags::Ion>::value);

        using SpeciesAtomicPhysicsConfigType = particles::atomicPhysics::traits::GetParticleType_t<FrameType>;

        static constexpr uint16_t value = SpeciesAtomicPhysicsConfigType::numberAtomicStates;
    };
} // namespace picongpu::traits

/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ParticleType.hpp"

#include <pmacc/particles/memory/frames/Frame.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <cstdint>

namespace picongpu::traits
{
    /** compile time functor for accessing instantiated atomicData type in
     *  numberAtomicStates flag of species
     *
     * @tparam T_IonSpecies resolved typename of species with flag
     * @returns return value contained in ::type
     */
    template<typename T_IonSpecies>
    struct GetAtomicDataType
    {
        using FrameType = typename T_IonSpecies::FrameType;

        /* throw static assert if species lacks flag */
        PMACC_CASSERT_MSG(
            Species_missing_atomicDataType_flag,
            particles::atomicPhysics::traits::IsParticleType<
                particles::atomicPhysics::traits::GetParticleType_t<FrameType>,
                particles::atomicPhysics::Tags::Ion>::value);

        using SpeciesAtomicPhysicsConfigType = particles::atomicPhysics::traits::GetParticleType_t<FrameType>;

        using type = typename SpeciesAtomicPhysicsConfigType::AtomicDataType;
    };
} // namespace picongpu::traits

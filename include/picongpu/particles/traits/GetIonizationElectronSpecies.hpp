/* Copyright 2023 Brian Marre
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ParticleType.hpp"

#include <pmacc/particles/memory/frames/Frame.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <cstdint>

namespace picongpu::traits
{
    /** @class compile time functor for accessing electron species to be created by
     * ionizing transitions of this ion species
     *
     * @tparam T_IonSpecies resolved typename of species with flag
     * @returns return value contained in ::type
     */
    template<typename T_IonSpecies>
    struct GetIonizationElectronSpecies
    {
        using IonFrameType = typename T_IonSpecies::FrameType;

        /* throw static assert if ion species lacks flag */
        PMACC_CASSERT_MSG(
            Species_missing_ionizationElectronSpecies_flag,
            particles::atomicPhysics::traits::IsParticleType<
                particles::atomicPhysics::traits::GetParticleType_t<IonFrameType>,
                particles::atomicPhysics::Tags::Ion>::value);

        using SpeciesAtomicPhysicsConfigType = particles::atomicPhysics::traits::GetParticleType_t<IonFrameType>;

        using type = typename SpeciesAtomicPhysicsConfigType::IonizationElectronSpecies;

        // sanity check
        //      check that ionization electron species is actually flagged as electron,
        //       i.e. be binned into electron histogram
        using ElectronFrameType = typename SpeciesAtomicPhysicsConfigType::IonizationElectronSpecies::FrameType;
        PMACC_CASSERT_MSG(
            ionization_electron_species_not_marked_as_atomic_physics_electron,
            particles::atomicPhysics::traits::IsParticleType<
                particles::atomicPhysics::traits::GetParticleType_t<ElectronFrameType>,
                particles::atomicPhysics::Tags::Electron>::value);
    };
} // namespace picongpu::traits

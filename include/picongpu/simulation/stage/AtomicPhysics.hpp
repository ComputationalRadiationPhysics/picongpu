/* Copyright 2022-2023 Brian Marre
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

#include "picongpu/particles/atomicPhysics/ParticleType.hpp"

#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <cstdint>
#include <string>

namespace picongpu::simulation::stage
{
    /** public interface of AtomicPhysics stage
     *
     * @note indirection necessary to avoid always compiling atomicPhysics stages
     */
    struct AtomicPhysics
    {
        using SpeciesRepresentingAtomicPhysicsIons = particles::atomicPhysics::traits::
            FilterByParticleType_t<VectorAllSpecies, picongpu::particles::atomicPhysics::Tags::Ion>;
        auto static constexpr numberAtomicPhysicsIonSpecies
            = pmacc::mp_size<SpeciesRepresentingAtomicPhysicsIons>::value;

        // check at least one electron species defined if atomicPhyiscs is active
        using SpeciesRepresentingAtomicPhysicsElectrons = particles::atomicPhysics::traits::
            FilterByParticleType_t<VectorAllSpecies, picongpu::particles::atomicPhysics::Tags::Electron>;
        auto static constexpr numberAtomicPhysicsElectronSpecies
            = pmacc::mp_size<SpeciesRepresentingAtomicPhysicsElectrons>::value;

        static bool constexpr atomicPhysicsActive
            = (numberAtomicPhysicsIonSpecies > 0 && numberAtomicPhysicsElectronSpecies > 0);

        PMACC_CASSERT_MSG(
            at_least_one_species_marked_as_atomic_physics_electron_species_required,
            (numberAtomicPhysicsIonSpecies == 0) || (numberAtomicPhysicsElectronSpecies > 0));

    private:
        /** load the atomic input files for each species with atomicData
         *
         * create an atomicData data base object for each atomicPhysics ion species and stores them in the data
         * connector
         *
         * @todo allow reuse of atomicData dataBase objects in between species, Brian Marre, 2022
         */
        void loadAtomicInputData(DataConnector& dataConnector);

    public:
        //! @details indirection necessary to prevent compiling atomicPhysics kernels if no atomicPhysics species exist
        AtomicPhysics(picongpu::MappingDesc const mappingDesc);

        void fixAtomicStateInit(picongpu::MappingDesc const mappingDesc);

        void operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep) const;
    };
} // namespace picongpu::simulation::stage

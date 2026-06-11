/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ParticleType.hpp"
#include "picongpu/particles/param.hpp"

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
        static constexpr auto numberAtomicPhysicsIonSpecies
            = pmacc::mp_size<SpeciesRepresentingAtomicPhysicsIons>::value;

        // check at least one electron species defined if atomicPhyiscs is active
        using SpeciesRepresentingAtomicPhysicsElectrons = particles::atomicPhysics::traits::
            FilterByParticleType_t<VectorAllSpecies, picongpu::particles::atomicPhysics::Tags::Electron>;
        static constexpr auto numberAtomicPhysicsElectronSpecies
            = pmacc::mp_size<SpeciesRepresentingAtomicPhysicsElectrons>::value;

        static constexpr bool atomicPhysicsActive
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

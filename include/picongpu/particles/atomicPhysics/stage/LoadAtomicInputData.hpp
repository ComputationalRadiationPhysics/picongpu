/*
 * SPDX-FileCopyrightText: Brian Marre, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ParticleType.hpp"
#include "picongpu/particles/atomicPhysics/debug/PrintAtomicDataToConsole.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"
#include "picongpu/particles/traits/GetNumberAtomicStates.hpp"

#include <pmacc/particles/memory/frames/Frame.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/HasIdentifier.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <memory>
#include <stdexcept>
#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    /** pre-simulation stage for loading the user provided atomic input data
     *
     * @tparam T_IonSpecies species for which to call the functor
     */
    template<typename T_IonSpecies>
    struct LoadAtomicInputData
    {
        HINLINE void operator()(DataConnector& dataConnector) const
        {
            // might be alias, from here on out no more
            //! resolved type of alias T_IonSpecies
            using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

            /// @todo move to trait functor?, Brian Marre, 2022
            using FrameType = typename IonSpecies::FrameType;

            // get atomicData dataBase type
            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;

            // get species atomicPhysics config
            PMACC_CASSERT_MSG(
                Species_not_marked_as_atomic_physics_ion_species,
                traits::IsParticleType<traits::GetParticleType_t<FrameType>, Tags::Ion>::value);

            using SpeciesAtomicPhysicsConfigType = particles::atomicPhysics::traits::GetParticleType_t<FrameType>;

            constexpr char const* chargeStatesFileName = SpeciesAtomicPhysicsConfigType::chargeStatesFileName;
            constexpr char const* atomicStatesFileName = SpeciesAtomicPhysicsConfigType::atomicStatesFileName;
            constexpr char const* ipdIonizationStatesFileName
                = SpeciesAtomicPhysicsConfigType::ipdIonizationStatesFileName;

            constexpr char const* boundBoundFileName = SpeciesAtomicPhysicsConfigType::boundBoundTransitionsFileName;
            constexpr char const* boundFreeFileName = SpeciesAtomicPhysicsConfigType::boundFreeTransitionsFileName;
            constexpr char const* autonomousFileName = SpeciesAtomicPhysicsConfigType::autonomousTransitionsFileName;

            static_assert(
                pmacc::traits::HasIdentifiers<
                    typename IonSpecies::FrameType,
                    MakeSeq_t<
                        atomicStateCollectionIndex,
                        processClass,
                        transitionIndex,
                        binIndex,
                        accepted,
                        boundElectrons,
                        weighting,
                        momentum>>::type::value,
                "atomic physics: species is missing one of the following attributes: atomicStateCollectionIndex, "
                "processClass, "
                "transitionIndex, binIndex, accepted, boundElectrons, weighting, momentum");

            auto atomicData = std::make_unique<AtomicDataType>(
                std::string(chargeStatesFileName),
                std::string(atomicStatesFileName),
                std::string(ipdIonizationStatesFileName),
                std::string(boundBoundFileName),
                std::string(boundFreeFileName),
                std::string(autonomousFileName),
                // name of species
                FrameType::getName());

            if constexpr(picongpu::atomicPhysics::debug::atomicData::PRINT_TO_CONSOLE)
                // debug print of atomic data summary to stdout
                atomicData = particles::atomicPhysics::debug::printAtomicDataToConsole<
                    AtomicDataType,
                    true, // print summary standard ordered transitions
                    true // print summary inverse ordered transitions
                    >(std::move(atomicData));

            // cross check number of atomic states in inputData with species flag number of atomic states
            constexpr uint16_t numberAtomicStatesOfSpecies
                = picongpu::traits::GetNumberAtomicStates<IonSpecies>::value;

            if(numberAtomicStatesOfSpecies != static_cast<uint16_t>(atomicData->getNumberAtomicStates()))
            {
                throw std::runtime_error(
                    "atomicPhysics ERROR: numberAtomicStates flag and number of atomic states in "
                    "input file do not match");
            }

            dataConnector.consume(std::move(atomicData));
        }
    };

} // namespace picongpu::particles::atomicPhysics::stage

/* Copyright 2022-2023 Brian Marre, Rene Widera
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

#include "picongpu/simulation_defines.hpp" // need: picongpu/param/atomicPhysics_Debug.param

#include "picongpu/particles/atomicPhysics/ParticleType.hpp"
#include "picongpu/particles/atomicPhysics/debug/PrintAtomicDataToConsole.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"
#include "picongpu/particles/traits/GetNumberAtomicStates.hpp"

#include <pmacc/particles/memory/frames/Frame.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/GetFlagType.hpp>
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
                HasFlag<FrameType, atomicPhysics_<atomicPhysics::particleType::Ion<>>>::type::value == true);
            using AliasAtomicPhysicsFlag = typename GetFlagType<FrameType, atomicPhysics_<>>::type;
            using SpeciesAtomicPhysicsConfigType = typename pmacc::traits::Resolve<AliasAtomicPhysicsFlag>::type;

            constexpr char const* chargeStatesFileName = SpeciesAtomicPhysicsConfigType::chargeStatesFileName;
            constexpr char const* atomicStatesFileName = SpeciesAtomicPhysicsConfigType::atomicStateFileName;
            constexpr char const* pressureIonizationStatesFileName
                = SpeciesAtomicPhysicsConfigType::pressureIonizationFileName;

            constexpr char const* boundBoundFileName = SpeciesAtomicPhysicsConfigType::boundBoundTransitionsFileName;
            constexpr char const* boundFreeFileName = SpeciesAtomicPhysicsConfigType::boundFreeTransitionsFileName;
            constexpr char const* autonomousFileName = SpeciesAtomicPhysicsConfigType::autonomousTransitionsFileName;

            auto atomicData = std::make_unique<AtomicDataType>(
                std::string(chargeStatesFileName),
                std::string(atomicStatesFileName),
                std::string(pressureIonizationStatesFileName),
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
                throw std::runtime_error("atomicPhysics ERROR: numberAtomicStates flag and number of atomic states in "
                                         "input file do not match");
            }

            dataConnector.consume(std::move(atomicData));
        }
    };

} // namespace picongpu::particles::atomicPhysics::stage

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

#include "picongpu/simulation_defines.hpp" // need: picongpu/param/atomicPhysics2_Debug.param

#include "picongpu/particles/atomicPhysics2/DebugHelper.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"
#include "picongpu/particles/traits/GetNumberAtomicStates.hpp"

#include <pmacc/particles/memory/frames/Frame.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <memory>
#include <stdexcept>
#include <string>


namespace picongpu::particles::atomicPhysics2::stage
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

            //      get charge states input file name
            PMACC_CASSERT_MSG(
                Species_missing_charge_states_file_name_flag,
                HasFlag<FrameType, chargeStatesFileName<>>::type::value == true);
            using AliasChargeStatesFileName = typename GetFlagType<FrameType, chargeStatesFileName<>>::type;
            using ChargeStatesFileName = typename pmacc::traits::Resolve<AliasChargeStatesFileName>::type;

            //      get atomic states input file name
            PMACC_CASSERT_MSG(
                Species_missing_atomic_states_file_name_flag,
                HasFlag<FrameType, atomicStatesFileName<>>::type::value == true);
            using AliasAtomicStatesFileName = typename GetFlagType<FrameType, atomicStatesFileName<>>::type;
            using AtomicStatesFileName = typename pmacc::traits::Resolve<AliasAtomicStatesFileName>::type;

            //      get pressureIonization input file name
            PMACC_CASSERT_MSG(
                Species_missing_pressure_ionization_file_name_flag,
                HasFlag<FrameType, pressureIonizationStatesFileName<>>::type::value == true);
            using AliasPressureIonizationFileName =
                typename GetFlagType<FrameType, pressureIonizationStatesFileName<>>::type;
            using PressureIonizationFileName = typename pmacc::traits::Resolve<AliasPressureIonizationFileName>::type;

            //      get bound-bound transitions input file name
            PMACC_CASSERT_MSG(
                Species_missing_bound_bound_transitions_file_name_flag,
                HasFlag<FrameType, boundBoundTransitionsFileName<>>::type::value == true);
            using AliasBoundBoundFileName = typename GetFlagType<FrameType, boundBoundTransitionsFileName<>>::type;
            using BoundBoundFileName = typename pmacc::traits::Resolve<AliasBoundBoundFileName>::type;

            //      get bound-free transitions input file name
            PMACC_CASSERT_MSG(
                Species_missing_bound_free_transitions_file_name_flag,
                HasFlag<FrameType, boundFreeTransitionsFileName<>>::type::value == true);
            using AliasBoundFreeFileName = typename GetFlagType<FrameType, boundFreeTransitionsFileName<>>::type;
            using BoundFreeFileName = typename pmacc::traits::Resolve<AliasBoundFreeFileName>::type;

            //      get autonomous transitions input file name
            PMACC_CASSERT_MSG(
                Species_missing_autonomous_transitions_file_name_flag,
                HasFlag<FrameType, autonomousTransitionsFileName<>>::type::value == true);
            using AliasAutonomousFileName = typename GetFlagType<FrameType, autonomousTransitionsFileName<>>::type;
            using AutonomousFileName = typename pmacc::traits::Resolve<AliasAutonomousFileName>::type;

            auto atomicData = std::make_unique<AtomicDataType>(
                ChargeStatesFileName::str(),
                AtomicStatesFileName::str(),
                PressureIonizationFileName::str(),
                BoundBoundFileName::str(),
                BoundFreeFileName::str(),
                AutonomousFileName::str(),
                FrameType::getName()); // name of species

            if constexpr(picongpu::atomicPhysics2::debug::atomicData::PRINT_TO_CONSOLE)
                // debug print of atomic data summary to stdout
                atomicData = particles::atomicPhysics2::debug::printAtomicDataToConsole<
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

} // namespace picongpu::particles::atomicPhysics2::stage

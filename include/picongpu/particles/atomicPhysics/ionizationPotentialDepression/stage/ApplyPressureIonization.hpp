/* Copyright 2024 Brian Marre
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

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/LocalIPDInputFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/kernel/ApplyPressureIonization.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalFoundUnboundIonField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalTimeRemainingField.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"
#include "picongpu/particles/traits/GetIonizationElectronSpecies.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stage
{
    //! short hand for IPD namespace
    namespace s_IPD = picongpu::particles::atomicPhysics::ionizationPotentialDepression;

    /** IPD sub-stage for performing ApplyPressureIonization kernel call for one Ion Species for the Stewart-Pyatt
     *  ionization potential depression model
     *
     * @todo implement version for non atomicPhysics data species
     *
     * @tparam ion species with atomic data
     */
    template<typename T_IonSpecies, typename T_IPDModel>
    struct ApplyPressureIonization
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_ParticleSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            //! resolved type of electron species to spawn upon ionization
            using IonizationElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<
                VectorAllSpecies,
                typename picongpu::traits::GetIonizationElectronSpecies<T_IonSpecies>::type>;

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<T_IonSpecies>::type;

            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localTimeRemainingField
                = *dc.get<atomicPhysics::localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>>(
                    "LocalTimeRemainingField");
            auto& localFoundUnboundIonField
                = *dc.get<atomicPhysics::localHelperFields::LocalFoundUnboundIonField<picongpu::MappingDesc>>(
                    "LocalFoundUnboundIonField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());
            auto& electrons = *dc.get<IonizationElectronSpecies>(IonizationElectronSpecies::FrameType::getName());

            auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            // ipd input fields
            //{
            auto& localDebyeLengthField
                = *dc.get<s_IPD::localHelperFields::LocalDebyeLengthField<picongpu::MappingDesc>>(
                    "LocalDebyeLengthField");
            auto& localTemperatureEnergyField
                = *dc.get<s_IPD::localHelperFields::LocalTemperatureEnergyField<picongpu::MappingDesc>>(
                    "LocalTemperatureEnergyField");
            auto& localZStarField
                = *dc.get<s_IPD::localHelperFields::LocalZStarField<picongpu::MappingDesc>>("LocalZStarField");
            //}

            // macro for call of kernel on every superCell, see pull request #4321
            PMACC_LOCKSTEP_KERNEL(s_IPD::kernel::ApplyPressureIonizationKernel<T_IPDModel>())
                .config(mapper.getGridDim(), ions)(
                    mapper,
                    ions.getDeviceParticlesBox(),
                    electrons.getDeviceParticlesBox(),
                    localTimeRemainingField.getDeviceDataBox(),
                    localFoundUnboundIonField.getDeviceDataBox(),
                    atomicData.template getChargeStateDataDataBox</*on device*/ false>(),
                    atomicData.template getAtomicStateDataDataBox</*on device*/ false>(),
                    atomicData.template getPressureIonizationStateDataBox</*on device*/ false>(),
                    localDebyeLengthField.getDeviceDataBox(),
                    localTemperatureEnergyField.getDeviceDataBox(),
                    localZStarField.getDeviceDataBox());

            // no need to call fillAllGaps, since we do not leave any gaps

            // debug call
            if constexpr(picongpu::atomicPhysics::debug::kernel::applyPressureIonization::
                             ELECTRON_PARTICLE_BOX_FILL_GAPS)
                electrons.fillAllGaps();
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stage

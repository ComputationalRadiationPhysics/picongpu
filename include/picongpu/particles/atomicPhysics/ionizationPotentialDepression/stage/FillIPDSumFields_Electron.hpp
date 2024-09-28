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

/** @file FillIPDSumFields ionization potential depression(IPD) sub-stage for an electron species
 *
 * implements filling of IPD sum fields from reduction of all macro particles of the specified **electron** species
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/kernel/FillIPDSumFields_Electron.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalTimeRemainingField.hpp"

#include <string>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stage
{
    //! short hand for IPD namespace
    namespace s_IPD = picongpu::particles::atomicPhysics::ionizationPotentialDepression;

    /** IPD sub-stage for filling the sum fields required for calculating the IPD inputs for an ion species
     *
     * @tparam T_ElectronSpecies electron species to fill into sum fields
     * @tparam T_TemperatureFunctional functional to use for temperature calculation
     */
    template<typename T_ElectronSpecies, typename T_TemperatureFunctional>
    struct FillIPDSumFields_Electron
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_ParticleSpecies
        using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ElectronSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            static_assert(
                pmacc::traits::HasIdentifiers<typename ElectronSpecies::FrameType, MakeSeq_t<weighting, momentum>>::
                    type::value,
                "atomic physics: species is missing one of the following attributes: weighting, momentum");

            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localTimeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>>(
                "LocalTimeRemainingField");

            // pointer to memory, we will only work on device, no sync required
            // init pointer to particles and localSumFields
            auto& electrons = *dc.get<ElectronSpecies>(ElectronSpecies::FrameType::getName());

            auto& localSumWeightAllField
                = *dc.get<s_IPD::localHelperFields::SumWeightAllField<picongpu::MappingDesc>>("SumWeightAllField");
            auto& localSumTemperatureFunctionalField
                = *dc.get<s_IPD::localHelperFields::SumTemperatureFunctionalField<picongpu::MappingDesc>>(
                    "SumTemperatureFunctionalField");

            auto& localSumWeightElectronField
                = *dc.get<s_IPD::localHelperFields::SumWeightElectronsField<picongpu::MappingDesc>>(
                    "SumWeightElectronsField");

            // macro for call of kernel on every superCell, see pull request #4321
            PMACC_LOCKSTEP_KERNEL(s_IPD::kernel::FillIPDSumFieldsKernel_Electron<T_TemperatureFunctional>())
                .config(mapper.getGridDim(), electrons)(
                    mapper,
                    localTimeRemainingField.getDeviceDataBox(),
                    electrons.getDeviceParticlesBox(),
                    localSumWeightAllField.getDeviceDataBox(),
                    localSumTemperatureFunctionalField.getDeviceDataBox(),
                    localSumWeightElectronField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stage

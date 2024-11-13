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

#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/LocalIPDInputFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/kernel/ApplyIPDIonization.kernel"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/ApplyIPDIonization.def"
#include "picongpu/particles/atomicPhysics/localHelperFields/FoundUnboundIonField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"
#include "picongpu/particles/traits/GetIonizationElectronSpecies.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <type_traits>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stage
{
    //! short hand for IPD namespace
    namespace s_IPD = picongpu::particles::atomicPhysics::ionizationPotentialDepression;

    template<typename T_IonSpecies, typename T_IPDModel>
    HINLINE void ApplyIPDIonization<T_IonSpecies, T_IPDModel>::operator()(
        picongpu::MappingDesc const mappingDesc) const
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_ParticleSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;
        //! resolved type of electron species to spawn upon ionization
        using IonizationElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<
            VectorAllSpecies,
            typename picongpu::traits::GetIonizationElectronSpecies<T_IonSpecies>::type>;

        using AtomicDataType = typename picongpu::traits::GetAtomicDataType<T_IonSpecies>::type;

        // full local domain, no guards
        pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
        pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

        auto& timeRemainingField
            = *dc.get<atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                "TimeRemainingField");
        auto& foundUnboundIonField
            = *dc.get<atomicPhysics::localHelperFields::FoundUnboundIonField<picongpu::MappingDesc>>(
                "FoundUnboundIonField");

        auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());
        auto& electrons = *dc.get<IonizationElectronSpecies>(IonizationElectronSpecies::FrameType::getName());

        auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

        // ipd input fields
        auto& debyeLengthField
            = *dc.get<s_IPD::localHelperFields::DebyeLengthField<picongpu::MappingDesc>>("DebyeLengthField");
        auto& temperatureEnergyField
            = *dc.get<s_IPD::localHelperFields::TemperatureEnergyField<picongpu::MappingDesc>>(
                "TemperatureEnergyField");
        auto& zStarField = *dc.get<s_IPD::localHelperFields::ZStarField<picongpu::MappingDesc>>("ZStarField");
        auto idProvider = dc.get<IdProvider>("globalId");

        auto& fieldE = *dc.get<FieldE>(FieldE::getName());

        // macro for call of kernel on every superCell, see pull request #4321
        PMACC_LOCKSTEP_KERNEL(s_IPD::kernel::ApplyIPDIonizationKernel<
                                  T_IPDModel,
                                  std::integral_constant<bool, AtomicDataType::switchFieldIonization>>())
            .config(mapper.getGridDim(), ions)(
                mapper,
                idProvider->getDeviceGenerator(),
                ions.getDeviceParticlesBox(),
                electrons.getDeviceParticlesBox(),
                timeRemainingField.getDeviceDataBox(),
                foundUnboundIonField.getDeviceDataBox(),
                atomicData.template getChargeStateDataDataBox</*on device*/ false>(),
                atomicData.template getAtomicStateDataDataBox</*on device*/ false>(),
                atomicData.template getIPDIonizationStateDataBox</*on device*/ false>(),
                fieldE.getDeviceDataBox(),
                debyeLengthField.getDeviceDataBox(),
                temperatureEnergyField.getDeviceDataBox(),
                zStarField.getDeviceDataBox());

        // no need to call fillAllGaps, since we do not leave any gaps

        // debug call
        if constexpr(picongpu::atomicPhysics::debug::kernel::applyIPDIonization::ELECTRON_PARTICLE_BOX_FILL_GAPS)
            electrons.fillAllGaps();
    }

} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stage

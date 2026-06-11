/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/LocalIPDInputFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/kernel/ApplyIPDIonization.kernel"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/ApplyIPDIonization.def"
#include "picongpu/particles/atomicPhysics/localHelperFields/FoundUnboundIonField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/atomicPhysics/spawnFromSourceSpeciesModules/NeverSkipSuperCells.hpp"
#include "picongpu/particles/atomicPhysics/spawnFromSourceSpeciesModules/SkipFinishedSuperCellsAtomicPhysics.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"
#include "picongpu/particles/traits/GetIonizationElectronSpecies.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <type_traits>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stage
{
    //! short hand for IPD namespace
    namespace s_IPD = picongpu::particles::atomicPhysics::ionizationPotentialDepression;

    template<typename T_IonSpecies, typename T_IPDModel, typename T_SkipFinishedSuperCell>
    HINLINE void ApplyIPDIonization<T_IonSpecies, T_IPDModel, T_SkipFinishedSuperCell>::operator()(
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
        auto& freeElectronDensityField
            = *dc.get<s_IPD::localHelperFields::ZStarField<picongpu::MappingDesc>>("FreeElectronDensityField");
        auto idProvider = dc.get<IdProvider>("globalId");

        auto& fieldE = *dc.get<FieldE>(FieldE::getName());

        /** @details must use if-constexpr since "apply" in PMACC_LOCKSTEP_KERNEL macro is only able to handle
         *      typenames, not typename aliases, ... for some reason. */
        if constexpr(T_SkipFinishedSuperCell::value)
        {
            PMACC_LOCKSTEP_KERNEL(
                s_IPD::kernel::ApplyIPDIonizationKernel<
                    particles::atomicPhysics::spawnFromSourceSpeciesModules::SkipFinishedSuperCellsAtomicPhysics,
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
                    zStarField.getDeviceDataBox(),
                    freeElectronDensityField.getDeviceDataBox());
        }
        else
        {
            PMACC_LOCKSTEP_KERNEL(
                s_IPD::kernel::ApplyIPDIonizationKernel<
                    particles::atomicPhysics::spawnFromSourceSpeciesModules::NeverSkipSuperCells,
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
                    zStarField.getDeviceDataBox(),
                    freeElectronDensityField.getDeviceDataBox());
        }

        // no need to call fillAllGaps, since we do not leave any gaps

        // debug call
        if constexpr(picongpu::atomicPhysics::debug::kernel::applyIPDIonization::ELECTRON_PARTICLE_BOX_FILL_GAPS)
            electrons.fillAllGaps();
    }

} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stage

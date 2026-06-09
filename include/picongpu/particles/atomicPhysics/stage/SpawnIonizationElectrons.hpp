/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2023-2024 Brian Marre
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

/** @file recordChanges sub-stage of atomicPhysics
 *
 * - create ionization macro electrons for ionizing transitions
 * - add delta energy of transition to interaction histogram bin for collisional transitions
 * - update atomic configNumber
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClassGroup.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"
#include "picongpu/particles/atomicPhysics/kernel/SpawnIonizationMacroElectrons.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/traits/GetIonizationElectronSpecies.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::stage
{
    namespace enums = picongpu::particles::atomicPhysics::enums;

    /** spawn ionization electrons for all macro ions that choose ionizing atomic physics processes
     *
     * @tparam T_checkForAccepted false =^= process all macro ions independent of their accepted_ attribute value,
     *  true =^= only process macro particles that are marked as accepted_==true and reset their accepted_ attribute to
     *  false afterwards
     */
    template<typename T_IonSpecies, typename T_checkForAccepted>
    struct SpawnIonizationElectrons
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! resolved type of electron species to spawn upon ionization
        using IonizationElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<
            VectorAllSpecies,
            typename picongpu::traits::GetIonizationElectronSpecies<IonSpecies>::type>;

        using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;

        using DistributionFloat = pmacc::random::distributions::Uniform<float_X>;
        using RngFactoryFloat = particles::functor::misc::Rng<DistributionFloat>;

        using IPDModel = picongpu::atomicPhysics::IPDModel;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc, uint32_t currentStep) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& timeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                "TimeRemainingField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());
            auto& electrons = *dc.get<IonizationElectronSpecies>(IonizationElectronSpecies::FrameType::getName());
            auto idProvider = dc.get<IdProvider>("globalId");

            auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            // spawn ionization electrons
            //      bound-free based transitions
            if constexpr(AtomicDataType::switchElectronicIonization || AtomicDataType::switchFieldIonization)
            {
                using SpawnElectrons_BoundFree
                    = picongpu::particles::atomicPhysics::kernel::SpawnIonizationMacroElectronsKernel<
                        IPDModel,
                        enums::ProcessClassGroup::boundFreeBased,
                        T_checkForAccepted::value>;

                // spawn ionization electrons for bound-free based processes
                PMACC_LOCKSTEP_KERNEL(SpawnElectrons_BoundFree())
                    .config(mapper.getGridDim(), ions)(
                        mapper,
                        idProvider->getDeviceGenerator(),
                        timeRemainingField.getDeviceDataBox(),
                        ions.getDeviceParticlesBox(),
                        electrons.getDeviceParticlesBox(),
                        atomicData.template getAtomicStateDataDataBox<false>(),
                        atomicData
                            .template getBoundFreeTransitionDataBox<false, enums::TransitionOrdering::byLowerState>());
            }

            //      autonomous based transitions
            if constexpr(AtomicDataType::switchAutonomousIonization)
            {
                using SpawnElectrons_Autonomous
                    = picongpu::particles::atomicPhysics::kernel::SpawnIonizationMacroElectronsKernel<
                        IPDModel,
                        enums::ProcessClassGroup::autonomousBased,
                        T_checkForAccepted::value>;

                RngFactoryFloat rngFactory = RngFactoryFloat{currentStep};

                IPDModel::template callKernelWithIPDInput<SpawnElectrons_Autonomous, IonSpecies::FrameType::frameSize>(
                    dc,
                    mapper,
                    idProvider->getDeviceGenerator(),
                    timeRemainingField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox(),
                    electrons.getDeviceParticlesBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData
                        .template getAutonomousTransitionDataBox<false, enums::TransitionOrdering::byUpperState>(),
                    rngFactory,
                    atomicData.template getChargeStateDataDataBox<false>());
            }

            // no need to call fillAllGaps, since we do not leave any gaps

            // debug call
            if constexpr(picongpu::atomicPhysics::debug::kernel::spawnIonizationElectrons::
                             ELECTRON_PARTICLE_BOX_FILL_GAPS)
                electrons.fillAllGaps();
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage

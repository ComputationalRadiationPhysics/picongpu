/* Copyright 2023 Brian Marre
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

//! @file record all accepted transition's suggested changes

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/kernel/RecordUsedElectronHistogramWeight.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalTimeRemainingField.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    /** atomicPhysics sub-stage recording for every accepted transition shared physics
     *  resource usage
     *
     * for example the histogram in weight usage of a collisional ionization,
     *  but not the ionization macro electron spawn, since that is not a shared resource.
     *
     * @attention assumes that the ChooseTransition, ExtractTransitionCollectionIndex
     *  and AcceptTransitionTest stages have been executed previously in the current
     *  atomicPhysics time step.
     *
     * @todo fuse kernels into AcceptTransitionTest kernel?, Brian Marre, 2023
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct RecordSuggestedChanges
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;

            auto& localTimeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>>(
                "LocalTimeRemainingField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            auto& localElectronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");

            // electronic collisional transition channel active
            if constexpr(
                AtomicDataType::switchElectronicExcitation || AtomicDataType::switchElectronicDeexcitation
                || AtomicDataType::switchElectronicIonization)
            {
                // macro for kernel call for every superCell
                PMACC_LOCKSTEP_KERNEL(
                    picongpu::particles::atomicPhysics::kernel::RecordUsedElectronHistogramWeightKernel<
                        picongpu::atomicPhysics::ElectronHistogram>())
                    .config(mapper.getGridDim(), ions)(
                        mapper,
                        localTimeRemainingField.getDeviceDataBox(),
                        ions.getDeviceParticlesBox(),
                        localElectronHistogramField.getDeviceDataBox());
            }

            /// @todo implement photonic collisional interactions, Brian Marre, 2023

            /// @todo implement field ionization field energy accounting, Brian Marre, 2023
        }
    };

} // namespace picongpu::particles::atomicPhysics::stage

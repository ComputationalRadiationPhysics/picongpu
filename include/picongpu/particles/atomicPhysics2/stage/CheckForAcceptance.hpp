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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file checkIonsForAcceptance sub-stage of atomicPhysics
 *
 * go over all macro-ions, check for each
 *  if they accepted a transitions, if not mark the super Cell in
 *  the AllMacroIonsAcceptedField as false
 */

#pragma once

#include "picongpu/particles/atomicPhysics2/kernel/CheckForAcceptance.kernel"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalTimeRemainingField.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics2::stage
{
    /** atomic physics sub-stage
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct CheckForAcceptance
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
            pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg<T_IonSpecies::FrameType::frameSize>();

            auto& localTimeRemainingField
                = *dc.get<picongpu::particles::atomicPhysics2::localHelperFields::LocalTimeRemainingField<
                    picongpu::MappingDesc>>("LocalTimeRemainingField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            auto& localAllIonsAcceptedField
                = *dc.get<picongpu::particles::atomicPhysics2::localHelperFields::
                              LocalElectronHistogramOverSubscribedField<picongpu::MappingDesc>>(
                    "LocalAllMacroIonsAcceptedField");

            // call kernel for each superCell
            PMACC_LOCKSTEP_KERNEL(picongpu::particles::atomicPhysics2::kernel::CheckForAcceptanceKernel(), workerCfg)
            (mapper.getGridDim())(
                mapper,
                localTimeRemainingField.getDeviceDataBox(),
                ions.getDeviceParticlesBox(),
                localAllIonsAcceptedField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stage

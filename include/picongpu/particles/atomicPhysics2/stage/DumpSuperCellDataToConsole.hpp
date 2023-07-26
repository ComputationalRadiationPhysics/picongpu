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

//! @file dump superCell atomicPhysics data to console, debug stage of atomicPhysics


#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/kernel/DumpSuperCellDataToConsole.kernel"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::stage
{
    /** @class atomicPhysics sub-stage dumping all superCell general atomicPhysics data to console,
     * calls the corresponding kernel per superCell
     *
     * is called once per time step for the entire local simulation volume by the atomicPhysicsStage
     */
    struct DumpSuperCellDataToConsole
    {
        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg(MappingDesc::SuperCellSize{});

            auto& localRejectionProbabilityCacheField
                = *dc.get<picongpu::particles::atomicPhysics2::localHelperFields::LocalRejectionProbabilityCacheField<
                    picongpu::MappingDesc>>("LocalRejectionProbabilityCacheField");

            auto& localElectronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics2::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics2::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");

            auto& localTimeStepField = *dc.get<
                picongpu::particles::atomicPhysics2::localHelperFields::LocalTimeStepField<picongpu::MappingDesc>>(
                "LocalTimeStepField");

            auto& localTimeRemainingField
                = *dc.get<picongpu::particles::atomicPhysics2::localHelperFields::LocalTimeRemainingField<
                    picongpu::MappingDesc>>("LocalTimeRemainingField");


            using DumpToConsole = picongpu::particles::atomicPhysics2::kernel::DumpSuperCellDataToConsoleKernel;

            PMACC_LOCKSTEP_KERNEL(DumpToConsole(), workerCfg)
            (mapper.getGridDim())(
                mapper,
                localElectronHistogramField.getDeviceDataBox(),
                localRejectionProbabilityCacheField.getDeviceDataBox(),
                localTimeStepField.getDeviceDataBox(),
                localTimeRemainingField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stage

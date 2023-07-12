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

//! @file dump superCell atomicPhysics data to console, debug stage of atomicPhysics


#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/kernel/DumpSuperCellDataToConsole.kernel"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics2::stage
{
    /** @class atomicPhysics sub-stage dumping an atomicPhysics superCellField to console,
     * calls the corresponding kernel per superCell
     *
     * is called once per time step for the entire local simulation volume by the atomicPhysics stage
     */
    template<typename T_FieldType, typename T_PrintFunctor>
    struct DumpSuperCellDataToConsole
    {
        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc, std::string const superCellFieldName) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg<1u>();

            T_FieldType& superCellField = *dc.get<T_FieldType>(superCellFieldName);

            using DumpToConsole
                = picongpu::particles::atomicPhysics2::kernel::DumpSuperCellDataToConsoleKernel<T_PrintFunctor>;

            PMACC_LOCKSTEP_KERNEL(DumpToConsole(), workerCfg)
            (mapper.getGridDim())(mapper, superCellField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stage

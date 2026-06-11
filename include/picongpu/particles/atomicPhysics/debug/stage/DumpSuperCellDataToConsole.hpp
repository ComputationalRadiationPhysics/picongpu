/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file dump superCell atomicPhysics data to console, debug stage of atomicPhysics


#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/debug/kernel/DumpSuperCellDataToConsole.kernel"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::stage
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

            T_FieldType& superCellField = *dc.get<T_FieldType>(superCellFieldName);

            using DumpToConsole
                = picongpu::particles::atomicPhysics::kernel::DumpSuperCellDataToConsoleKernel<T_PrintFunctor>;

            PMACC_LOCKSTEP_KERNEL(DumpToConsole())
                .template config<1u>(mapper.getGridDim())(mapper, superCellField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage

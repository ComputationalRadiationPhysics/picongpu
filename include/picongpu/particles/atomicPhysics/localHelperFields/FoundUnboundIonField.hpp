/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file implements bool storage superCellField if an unbound ion was found previously

#pragma once

#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /** debug only, write foundUnboundionField to console
     *
     * @attention only creates output if atomicPhysics debug setting CPU_OUTPUT_ACTIVE == True
     * @attention only useful if compiling for serial or cpu backend, otherwise will throw compile error if called by
     *  DumpSuperCellDataToConsole kernel on device
     */
    struct PrintFoundUnboundToConsole
    {
        //! cpu version
        template<typename T_Acc>
        HDINLINE auto operator()(
            T_Acc const&,
            uint32_t const foundUnbound,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
            if(foundUnbound)
                printf("foundUnbound %s: True\n", superCellIdx.toString(",", "[]").c_str());
            else
                printf("foundUnbound %s: False\n", superCellIdx.toString(",", "[]").c_str());
        }

        //! gpu version, does nothing
        template<typename T_Acc>
        HDINLINE auto operator()(
            T_Acc const&,
            uint32_t const foundUnbound,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<!std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
        }
    };

    /**superCell field
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     */
    template<typename T_MappingDescription>
    struct FoundUnboundIonField : public SuperCellField<uint32_t, T_MappingDescription, /*no guards*/ false>
    {
        FoundUnboundIonField(T_MappingDescription const& mappingDesc)
            : SuperCellField<uint32_t, T_MappingDescription, /*no guards*/ false>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "FoundUnboundIonField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields

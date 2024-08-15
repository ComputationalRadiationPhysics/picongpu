/* Copyright 2024Brian Marre
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
    struct LocalFoundUnboundIonField : public SuperCellField<uint32_t, T_MappingDescription, /*no guards*/ false>
    {
        LocalFoundUnboundIonField(T_MappingDescription const& mappingDesc)
            : SuperCellField<uint32_t, T_MappingDescription, /*no guards*/ false>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "LocalFoundUnboundIonField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields

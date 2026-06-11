/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file implements bool storage superCellField if histogram overSubscribed

#pragma once

#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    //! debug only, write electronHistogramOverSubcribed to console
    struct PrintOverSubcriptionFieldToConsole
    {
        //! cpu version
        template<typename T_Acc>
        HDINLINE auto operator()(
            T_Acc const&,
            uint32_t const overSubscribed,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
            if(overSubscribed)
                printf("overSubscribed %s: True\n", superCellIdx.toString(",", "[]").c_str());
            else
                printf("overSubscribed %s: False\n", superCellIdx.toString(",", "[]").c_str());
        }

        //! gpu version, does nothing
        template<typename T_Acc>
        HDINLINE auto operator()(
            T_Acc const&,
            uint32_t const overSubscribed,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<!std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
        }
    };

    /** superCell field of the electronHistogram over subscribed state
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     */
    template<typename T_MappingDescription>
    struct SharedResourcesOverSubscribedField
        : public SuperCellField<uint32_t, T_MappingDescription, false /*no guards*/>
    {
        SharedResourcesOverSubscribedField(T_MappingDescription const& mappingDesc)
            : SuperCellField<uint32_t, T_MappingDescription, false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "SharedResourcesOverSubscribedField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields

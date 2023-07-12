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

//! @file implements bool storage superCellField if histogram overSubscribed

#pragma once

#include "picongpu/particles/atomicPhysics2/SuperCellField.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics2::localHelperFields
{
    /**@class superCell field of the electronHistogram over subscribed state
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     */
    template<typename T_MappingDescription>
    struct LocalElectronHistogramOverSubscribedField
        : public SuperCellField<uint32_t, T_MappingDescription, false /*no guards*/>
    {
        LocalElectronHistogramOverSubscribedField(T_MappingDescription const& mappingDesc)
            : SuperCellField<uint32_t, T_MappingDescription, false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "LocalElectronHistogramOverSubscribedField";
        }
    };
} // namespace picongpu::particles::atomicPhysics2::localHelperFields

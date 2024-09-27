/* Copyright 2024 Brian Marre
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

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::localHelperFields
{
    /**superCell field of local temperature * k_Boltzman
     *
     * unit: eV, not weighted
     *
     * @details required for calculating the local ionization potential depression(IPD) and filled by
     *  calculateIPDInput kernel.
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     */
    template<typename T_MappingDescription>
    struct LocalTemperatureEnergyField : public SuperCellField<float_X, T_MappingDescription, /*no guards*/ false>
    {
        LocalTemperatureEnergyField(T_MappingDescription const& mappingDesc)
            : SuperCellField<float_X, T_MappingDescription, /*no guards*/ false>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "LocalTemperatureEnergyField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::localHelperFields

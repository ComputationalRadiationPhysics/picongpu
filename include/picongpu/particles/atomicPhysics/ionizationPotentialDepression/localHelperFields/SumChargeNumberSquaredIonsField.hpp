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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::localHelperFields
{
    /**helper superCell field of the weighted sum of (charge/(e * weight))^2 of all macro particles of all ion species.
     *
     * @details weighted by macro particle weight /sim.unit.typicalNumParticlesPerMacroParticle()
     * @details unit: unitless * weight / sim.unit.typicalNumParticlesPerMacroParticle()
     *
     * @note required for calculating local z^* for ionization potential depression(IPD)
     * @note is used to keep intermediate results between kernel calls for different species
     *
     * @attention field value only valid after fillIPDSumFields kernel has been executed for **all** ion species.
     * @attention in units of picongpu::sim.unit.typicalNumParticlesPerMacroParticle()!
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     */
    template<typename T_MappingDescription>
    struct SumChargeNumberSquaredIonsField : public SuperCellField<float_X, T_MappingDescription, false /*no guards*/>
    {
        SumChargeNumberSquaredIonsField(T_MappingDescription const& mappingDesc)
            : SuperCellField<float_X, T_MappingDescription, false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "SumChargeNumberSquaredIonsField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::localHelperFields

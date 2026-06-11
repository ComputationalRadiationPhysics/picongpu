/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
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

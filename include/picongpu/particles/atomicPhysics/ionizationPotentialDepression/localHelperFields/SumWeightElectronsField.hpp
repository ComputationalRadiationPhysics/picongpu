/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::localHelperFields
{
    /**helper superCell field of sum of weight of all electron macro particles.
     *
     * @details unit: 1 / sim.unit.typicalNumParticlesPerMacroParticle()
     *
     * @note required for calculating local debye length for ionization potential depression(IPD)
     * @note is used to keep intermediate results between kernel calls for different species
     *
     * @attention field value only valid after fillIPDSumFields kernel has been executed for **all** electron species.
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     */
    template<typename T_MappingDescription>
    struct SumWeightElectronsField : public SuperCellField<float_X, T_MappingDescription, false /*no guards*/>
    {
        SumWeightElectronsField(T_MappingDescription const& mappingDesc)
            : SuperCellField<float_X, T_MappingDescription, false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "SumWeightElectronsField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::localHelperFields

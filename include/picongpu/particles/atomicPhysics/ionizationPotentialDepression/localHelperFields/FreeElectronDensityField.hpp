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
    /**superCell field of the local free electron density
     *
     * @details unitless, not weighted
     *
     * @note required for calculating the local ionization potential depression(IPD) and filled by
     *  calculateIPDInput kernel.
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     */
    template<typename T_MappingDescription>
    struct FreeElectronDensityField : public SuperCellField<float_X, T_MappingDescription, /*no guards*/ false>
    {
        FreeElectronDensityField(T_MappingDescription const& mappingDesc)
            : SuperCellField<float_X, T_MappingDescription, /*no guards*/ false>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "FreeElectronDensityField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::localHelperFields

/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file implements a super cell local cache of of each electron histogram bin's
 *   rejectionProbability due to over subscription
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/RejectionProbabilityCache_Bin.hpp"
#include "picongpu/particles/atomicPhysics/param.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /** superCell field of the rejectionProbabilityCache for all electron histogram bins
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     */
    template<typename T_MappingDescription>
    struct RejectionProbabilityCacheField_Bin
        : public SuperCellField<
              RejectionProbabilityCache_Bin<picongpu::atomicPhysics::ElectronHistogram::numberBins>,
              T_MappingDescription,
              false /*no guards*/>
    {
        using ElementType = RejectionProbabilityCache_Bin<picongpu::atomicPhysics::ElectronHistogram::numberBins>;

        RejectionProbabilityCacheField_Bin(T_MappingDescription const& mappingDesc)
            : SuperCellField<
                  RejectionProbabilityCache_Bin<picongpu::atomicPhysics::ElectronHistogram::numberBins>,
                  T_MappingDescription,
                  false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "RejectionProbabilityCacheField_Bin";
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields

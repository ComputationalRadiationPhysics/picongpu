/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2023-2024 Brian Marre
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

/** @file implements a super cell local cache of of each electron histogram bin's
 *   rejectionProbability due to over subscription
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/RejectionProbabilityCache_Cell.hpp"
#include "picongpu/particles/atomicPhysics/param.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /** superCell field of the rejectionProbabilityCache for all cells
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     */
    template<typename T_MappingDescription>
    struct RejectionProbabilityCacheField_Cell
        : public SuperCellField<
              RejectionProbabilityCache_Cell<pmacc::math::CT::volume<picongpu::SuperCellSize>::type::value>,
              T_MappingDescription,
              false /*no guards*/>
    {
        using ElementType
            = RejectionProbabilityCache_Cell<pmacc::math::CT::volume<picongpu::SuperCellSize>::type::value>;

        RejectionProbabilityCacheField_Cell(T_MappingDescription const& mappingDesc)
            : SuperCellField<
                  RejectionProbabilityCache_Cell<pmacc::math::CT::volume<picongpu::SuperCellSize>::type::value>,
                  T_MappingDescription,
                  false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "RejectionProbabilityCacheField_Cell";
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields

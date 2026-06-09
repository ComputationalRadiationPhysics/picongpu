/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2025 Brian Marre
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

// need simulation.param for normalisation and units, memory.param for SuperCellSize and dim.param for simDim
#include "picongpu/defines.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    // get IPD from IPD model
    template<typename T_IPDModel, typename T_AtomicStateDataDataBox>
    HDINLINE float_X getIPD(
        T_AtomicStateDataDataBox atomicStateBox,
        uint32_t const stateCollectionIndex,
        typename T_IPDModel::SuperCellConstantInput const superCellConstantIPDInput)
    {
        auto const stateConfigNumber = atomicStateBox.configNumber(stateCollectionIndex);
        uint8_t const stateChargeState = T_AtomicStateDataDataBox::ConfigNumber::getChargeState(stateConfigNumber);

        // eV
        return T_IPDModel::ipd(superCellConstantIPDInput, stateChargeState);
    }
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression

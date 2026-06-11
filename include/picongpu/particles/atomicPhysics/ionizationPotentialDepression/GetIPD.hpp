/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

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

//! @file implements functor for setting to atomicState

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <pmacc/static_assert.hpp>

namespace picongpu::particles::atomicPhysics2
{
    struct SetAtomicState
    {
        template<typename T_AtomicStateDataDataBox, typename T_Ion, typename T_CollectionIndex>
        DINLINE static void op(
            T_AtomicStateDataDataBox const atomicStateBox,
            T_Ion& ion,
            T_CollectionIndex const newAtomicStateCollectionIndex)
        {
            using ConfigNumber = typename T_AtomicStateDataDataBox::ConfigNumber;

            typename T_AtomicStateDataDataBox::Idx const newAtomicConfigNumber
                = atomicStateBox.configNumber(newAtomicStateCollectionIndex);

            PMACC_DEVICE_ASSERT_MSG(
                ConfigNumber::getBoundElectrons(newAtomicConfigNumber) <= ConfigNumber::atomicNumber,
                "Number of bound electrons must be <= Z");

            // update atomic State
            ion[atomicStateCollectionIndex_] = newAtomicStateCollectionIndex;
            // update charge State
            ion[boundElectrons_] = ConfigNumber::getBoundElectrons(newAtomicConfigNumber);
        }
    };
} // namespace picongpu::particles::atomicPhysics2

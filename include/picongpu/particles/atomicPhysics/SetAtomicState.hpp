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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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

#include <cstdint>

namespace picongpu::particles::atomicPhysics
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

        /** hard set of input to ions
         *
         * @attention no check for consistency of arguments! Use only if you know what you are doing!
         *
         * @param ion macro particle to set atomic state for
         * @param newNumberBoundElectrons value to set for boundElectrons particle attribute, new number of bound
         *  electrons of the ion after this call
         * @param newAtomicStateCollectionIndex value to set for collectionIndex of atomic state attribute, new atomic
         *  state of the ion after this call
         */
        template<typename T_Ion, typename T_CollectionIndex>
        DINLINE static void hard(
            T_Ion& ion,
            uint8_t const newNumberBoundElectrons,
            T_CollectionIndex const newAtomicStateCollectionIndex)
        {
            // update atomic State
            ion[atomicStateCollectionIndex_] = newAtomicStateCollectionIndex;
            // update charge State
            ion[boundElectrons_] = newNumberBoundElectrons;
        }
    };
} // namespace picongpu::particles::atomicPhysics

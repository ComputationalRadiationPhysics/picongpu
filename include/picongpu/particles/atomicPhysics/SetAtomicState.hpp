/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file implements functor for setting to atomicState

#pragma once

#include "picongpu/defines.hpp"

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

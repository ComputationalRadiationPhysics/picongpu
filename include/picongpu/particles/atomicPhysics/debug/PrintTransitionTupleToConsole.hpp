/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/GetStateFromTransitionTuple.hpp"

#include <cstdint>
#include <iostream>
#include <string>

namespace picongpu::particles::atomicPhysics::debug
{
    //! debug only, write transition tuple to console
    template<typename T_Tuple, typename T_Idx, typename T_Value, typename T_ConfigNumber>
    ALPAKA_FN_HOST void printTransitionTupleToConsole(T_Tuple const& tuple)
    {
        T_Idx const upperAtomicState
            = picongpu::particles::atomicPhysics::atomicData::getUpperStateConfigNumber<T_Idx, T_Value>(tuple);
        T_Idx const lowerAtomicState
            = picongpu::particles::atomicPhysics::atomicData::getLowerStateConfigNumber<T_Idx, T_Value>(tuple);
        uint8_t const upperChargeState = T_ConfigNumber::getChargeState(upperAtomicState);
        uint8_t const lowerChargeState = T_ConfigNumber::getChargeState(lowerAtomicState);

        std::cout << "Transition: (" << static_cast<uint16_t>(lowerChargeState) << ": " << lowerAtomicState << ") -> ("
                  << static_cast<uint16_t>(upperChargeState) << ":" << upperAtomicState << ")" << std::endl;
    }
} // namespace picongpu::particles::atomicPhysics::debug

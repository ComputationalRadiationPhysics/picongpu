/* Copyright 2024 Brian Marre
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

        std::cout << "State : " << static_cast<uint16_t>(lowerChargeState) << ", " << lowerAtomicState << ", "
                  << static_cast<uint16_t>(upperChargeState) << ", " << upperAtomicState << std::endl;
    }
} // namespace picongpu::particles::atomicPhysics::debug

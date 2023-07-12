/* Copyright 2022-2023 Brian Marre
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/atomicData/AtomicTuples.def"

#include <tuple>

/** @file implements a unified getter for the upper and lower atomicState configNumbers from Transition Tuples
 */

namespace picongpu::particles::atomicPhysics2::atomicData
{
    template<typename T_ConfigNumberDataType, typename T_Value = float_X>
    HINLINE T_ConfigNumberDataType
    getLowerStateConfigNumber(atomicData::BoundBoundTransitionTuple<T_Value, T_ConfigNumberDataType> const& tupel)
    {
        return std::get<7>(tupel);
    }

    template<typename T_ConfigNumberDataType, typename T_Value = float_X>
    HINLINE T_ConfigNumberDataType
    getUpperStateConfigNumber(atomicData::BoundBoundTransitionTuple<T_Value, T_ConfigNumberDataType> const& tupel)
    {
        return std::get<8>(tupel);
    }

    template<typename T_ConfigNumberDataType, typename T_Value = float_X>
    HINLINE T_ConfigNumberDataType
    getLowerStateConfigNumber(atomicData::BoundFreeTransitionTuple<T_Value, T_ConfigNumberDataType> const& tupel)
    {
        return std::get<8>(tupel);
    }

    template<typename T_ConfigNumberDataType, typename T_Value = float_X>
    T_ConfigNumberDataType getUpperStateConfigNumber(
        atomicData::BoundFreeTransitionTuple<T_Value, T_ConfigNumberDataType> const& tupel)
    {
        return std::get<9>(tupel);
    }

    // T_Value only to keep interface consistent
    template<typename T_ConfigNumberDataType, typename T_Value = float_X>
    T_ConfigNumberDataType getLowerStateConfigNumber(
        atomicData::AutonomousTransitionTuple<T_ConfigNumberDataType> const& tupel)
    {
        return std::get<1>(tupel);
    }

    // T_Value only to keep interface consistent
    template<typename T_ConfigNumberDataType, typename T_Value = float_X>
    T_ConfigNumberDataType getUpperStateConfigNumber(
        atomicData::AutonomousTransitionTuple<T_ConfigNumberDataType> const& tupel)
    {
        return std::get<2>(tupel);
    }

} // namespace picongpu::particles::atomicPhysics2::atomicData

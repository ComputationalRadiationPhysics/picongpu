/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicTuples.def"

#include <tuple>

/** @file implements a unified getter for the upper and lower atomicState configNumbers from Transition Tuples
 */

namespace picongpu::particles::atomicPhysics::atomicData
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

} // namespace picongpu::particles::atomicPhysics::atomicData

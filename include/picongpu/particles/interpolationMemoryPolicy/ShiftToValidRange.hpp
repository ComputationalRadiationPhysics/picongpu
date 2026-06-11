/*
 * SPDX-FileCopyrightText: Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/math/math.hpp>

namespace picongpu
{
    namespace particles
    {
        namespace interpolationMemoryPolicy
        {
            /** Shift position to valid range [0,1)
             *  and repositions memory accordingly.
             *  This is necessary if a particle moves
             *  outside of its cell during a sub-stepping cycle
             *  Returns: shifted position and shifted memory. */
            struct ShiftToValidRange
            {
                template<typename T_MemoryType, typename T_PosType>
                HDINLINE T_MemoryType memory(T_MemoryType const& mem, T_PosType const& pos) const
                {
                    T_PosType const pos_floor = pmacc::math::floor(pos);
                    return mem.shift(precisionCast<int>(pos_floor));
                }

                template<typename T_PosType>
                HDINLINE T_PosType position(T_PosType const& pos) const
                {
                    T_PosType const pos_floor = pmacc::math::floor(pos);
                    return pos - pos_floor;
                }
            };

        } // namespace interpolationMemoryPolicy
    } // namespace particles
} // namespace picongpu

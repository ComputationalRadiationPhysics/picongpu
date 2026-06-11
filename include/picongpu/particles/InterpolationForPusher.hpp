/*
 * SPDX-FileCopyrightText: Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/Align.hpp>

namespace picongpu
{
    /** functor for particle field interpolator
     *
     * This functor is a simplification of the full
     * field to particle interpolator that can be used in the
     * particle pusher
     */
    template<typename T_Field2PartInt, typename T_MemoryType, typename T_FieldPosition>
    struct InterpolationForPusher
    {
        using Field2PartInt = T_Field2PartInt;

        HDINLINE
        InterpolationForPusher(T_MemoryType const& mem, T_FieldPosition const& fieldPos)
            : m_mem(mem)
            , m_fieldPos(fieldPos)
        {
        }

        /* apply shift policy before interpolation */
        template<typename T_PosType, typename T_ShiftPolicy>
        HDINLINE float3_X operator()(T_PosType const& pos, T_ShiftPolicy const& shiftPolicy) const
        {
            return Field2PartInt()(shiftPolicy.memory(m_mem, pos), shiftPolicy.position(pos), m_fieldPos);
        }

        /* interpolation using given memory and position */
        template<typename T_PosType>
        HDINLINE float3_X operator()(T_PosType const& pos) const
        {
            return Field2PartInt()(m_mem, pos, m_fieldPos);
        }


    private:
        PMACC_ALIGN(m_mem, T_MemoryType);
        PMACC_ALIGN(m_fieldPos, T_FieldPosition const);
    };

    /** functor to create particle field interpolator
     *
     * required to get interpolator for pusher
     */
    template<typename T_Field2PartInt>
    struct CreateInterpolationForPusher
    {
        template<typename T_MemoryType, typename T_FieldPosition>
        HDINLINE InterpolationForPusher<T_Field2PartInt, T_MemoryType, T_FieldPosition> operator()(
            T_MemoryType const& mem,
            T_FieldPosition const& fieldPos)
        {
            return InterpolationForPusher<T_Field2PartInt, T_MemoryType, T_FieldPosition>(mem, fieldPos);
        }
    };

} // namespace picongpu

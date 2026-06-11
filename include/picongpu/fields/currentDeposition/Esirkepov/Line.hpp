/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/types.hpp>

namespace picongpu
{
    namespace currentSolver
    {
        using namespace pmacc;

        template<typename T_Type>
        struct Line
        {
            using type = T_Type;

            type m_pos0;
            type m_pos1;

            DINLINE Line() = default;

            DINLINE Line(type const& pos0, type const& pos1) : m_pos0(pos0), m_pos1(pos1)
            {
            }

            DINLINE Line<type>& operator-=(type const& rhs)
            {
                m_pos0 -= rhs;
                m_pos1 -= rhs;
                return *this;
            }
        };

        template<typename T_Type>
        DINLINE Line<T_Type> operator-(Line<T_Type> const& lhs, T_Type const& rhs)
        {
            return Line<T_Type>(lhs.m_pos0 - rhs, lhs.m_pos1 - rhs);
        }

        template<typename T_Type>
        DINLINE Line<T_Type> operator-(T_Type const& lhs, Line<T_Type> const& rhs)
        {
            return Line<T_Type>(lhs - rhs.m_pos0, lhs - rhs.m_pos1);
        }

        /// auxillary function to rotate a vector

        template<int newXAxis, int newYAxis, int newZAxis>
        DINLINE float3_X rotateOrigin(float3_X const& vec)
        {
            return float3_X(vec[newXAxis], vec[newYAxis], vec[newZAxis]);
        }

        template<int newXAxis, int newYAxis>
        DINLINE float2_X rotateOrigin(float2_X const& vec)
        {
            return float2_X(vec[newXAxis], vec[newYAxis]);
        }

        /// auxillary function to rotate a line

        template<int newXAxis, int newYAxis, int newZAxis, typename T_Type>
        DINLINE Line<T_Type> rotateOrigin(Line<T_Type> const& line)
        {
            Line<T_Type> result(
                rotateOrigin<newXAxis, newYAxis, newZAxis>(line.m_pos0),
                rotateOrigin<newXAxis, newYAxis, newZAxis>(line.m_pos1));
            return result;
        }

        template<int newXAxis, int newYAxis, typename T_Type>
        DINLINE Line<T_Type> rotateOrigin(Line<T_Type> const& line)
        {
            Line<T_Type> result(
                rotateOrigin<newXAxis, newYAxis>(line.m_pos0),
                rotateOrigin<newXAxis, newYAxis>(line.m_pos1));
            return result;
        }

    } // namespace currentSolver

} // namespace picongpu

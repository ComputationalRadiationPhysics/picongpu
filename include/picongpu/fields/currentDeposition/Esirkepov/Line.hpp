/* Copyright 2013-2021 Heiko Burau, Rene Widera
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


#pragma once

#include "picongpu/simulation_defines.hpp"
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

            DINLINE Line()
            {
            }

            DINLINE Line(const type& pos0, const type& pos1) : m_pos0(pos0), m_pos1(pos1)
            {
            }

            DINLINE Line<type>& operator-=(const type& rhs)
            {
                m_pos0 -= rhs;
                m_pos1 -= rhs;
                return *this;
            }
        };

        template<typename T_Type>
        DINLINE Line<T_Type> operator-(const Line<T_Type>& lhs, const T_Type& rhs)
        {
            return Line<T_Type>(lhs.m_pos0 - rhs, lhs.m_pos1 - rhs);
        }

        template<typename T_Type>
        DINLINE Line<T_Type> operator-(const T_Type& lhs, const Line<T_Type>& rhs)
        {
            return Line<T_Type>(lhs - rhs.m_pos0, lhs - rhs.m_pos1);
        }

        /// auxillary function to rotate a vector

        template<int newXAxis, int newYAxis, int newZAxis>
        DINLINE float3_X rotateOrigin(const float3_X& vec)
        {
            return float3_X(vec[newXAxis], vec[newYAxis], vec[newZAxis]);
        }

        template<int newXAxis, int newYAxis>
        DINLINE float2_X rotateOrigin(const float2_X& vec)
        {
            return float2_X(vec[newXAxis], vec[newYAxis]);
        }
        /// auxillary function to rotate a line

        template<int newXAxis, int newYAxis, int newZAxis, typename T_Type>
        DINLINE Line<T_Type> rotateOrigin(const Line<T_Type>& line)
        {
            Line<T_Type> result(
                rotateOrigin<newXAxis, newYAxis, newZAxis>(line.m_pos0),
                rotateOrigin<newXAxis, newYAxis, newZAxis>(line.m_pos1));
            return result;
        }

        template<int newXAxis, int newYAxis, typename T_Type>
        DINLINE Line<T_Type> rotateOrigin(const Line<T_Type>& line)
        {
            Line<T_Type> result(
                rotateOrigin<newXAxis, newYAxis>(line.m_pos0),
                rotateOrigin<newXAxis, newYAxis>(line.m_pos1));
            return result;
        }

    } // namespace currentSolver

} // namespace picongpu

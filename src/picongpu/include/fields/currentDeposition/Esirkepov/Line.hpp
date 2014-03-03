/**
 * Copyright 2013 Heiko Burau, Rene Widera
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

#include "simulation_defines.hpp"
#include "types.h"

namespace picongpu
{
namespace currentSolverEsirkepov
{
using namespace PMacc;

template<typename T_Type>
struct Line
{
    typedef T_Type type;
    
    type pos0;
    type pos1;

    DINLINE Line(const type& pos0, const type & pos1) : pos0(pos0), pos1(pos1)
    {
    }

    DINLINE Line<type>& operator-=(const type & rhs)
    {
        pos0 -= rhs;
        pos1 -= rhs;
        return *this;
    }
};

template<typename T_Type>
DINLINE Line<T_Type> operator-(const Line<T_Type>& lhs, const T_Type& rhs)
{
    return Line<T_Type>(lhs.pos0 - rhs, lhs.pos1 - rhs);
}

template<typename T_Type>
DINLINE Line<T_Type> operator-(const T_Type& lhs, const Line<T_Type>& rhs)
{
    return Line<T_Type>(lhs - rhs.pos0, lhs - rhs.pos1);
}

///auxillary function to rotate a vector

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
///auxillary function to rotate a line

template<int newXAxis, int newYAxis, int newZAxis,typename T_Type>
DINLINE Line<T_Type> rotateOrigin(const Line<T_Type>& line)
{
    Line<T_Type> result(rotateOrigin<newXAxis, newYAxis, newZAxis > (line.pos0),
                rotateOrigin<newXAxis, newYAxis, newZAxis > (line.pos1));
    return result;
}

template<int newXAxis, int newYAxis,typename T_Type>
DINLINE Line<T_Type> rotateOrigin(const Line<T_Type>& line)
{
    Line<T_Type> result(rotateOrigin<newXAxis, newYAxis > (line.pos0),
                rotateOrigin<newXAxis, newYAxis > (line.pos1));
    return result;
}

} //namespace currentSolverEsirkepov

} //namespace picongpu


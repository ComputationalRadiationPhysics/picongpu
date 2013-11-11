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

struct Line
{
    float3_X pos0;
    float3_X pos1;

    DINLINE Line(const float3_X& pos0, const float3_X & pos1) : pos0(pos0), pos1(pos1)
    {
    }

    DINLINE Line& operator-=(const float3_X & rhs)
    {
        pos0 -= rhs;
        pos1 -= rhs;
        return *this;
    }
};

DINLINE Line operator-(const Line& lhs, const float3_X& rhs)
{
    return Line(lhs.pos0 - rhs, lhs.pos1 - rhs);
}

DINLINE Line operator-(const float3_X& lhs, const Line& rhs)
{
    return Line(lhs - rhs.pos0, lhs - rhs.pos1);
}

///auxillary function to rotate a vector

template<int newXAxis, int newYAxis, int newZAxis>
DINLINE float3_X rotateOrigin(const float3_X& vec)
{
    return float3_X(vec[newXAxis], vec[newYAxis], vec[newZAxis]);
}
///auxillary function to rotate a line

template<int newXAxis, int newYAxis, int newZAxis>
DINLINE Line rotateOrigin(const Line& line)
{
    Line result(rotateOrigin<newXAxis, newYAxis, newZAxis > (line.pos0),
                rotateOrigin<newXAxis, newYAxis, newZAxis > (line.pos1));
    return result;
}

} //namespace currentSolverEsirkepov

} //namespace picongpu


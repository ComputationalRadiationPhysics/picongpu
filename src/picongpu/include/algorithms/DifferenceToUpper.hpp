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

#include "types.h"
#include "math/vector/compile-time/Int.hpp"

namespace picongpu
{
template<unsigned T_Dim>
struct DifferenceToUpper;

template<>
struct DifferenceToUpper<DIM3>
{
    typedef PMacc::math::CT::Int< 1, 1, 1 > OffsetEnd;
    typedef PMacc::math::CT::Int< 0, 0, 0 > OffsetOrigin;

    template<class Memory >
    HDINLINE typename Memory::ValueType operator()(const Memory& mem, const uint32_t direction) const
    {
        const float_X reciWidth = float_X(1.0) / CELL_WIDTH;
        const float_X reciHeight = float_X(1.0) / CELL_HEIGHT;
        const float_X reciDepth = float_X(1.0) / CELL_DEPTH;
        switch (direction)
        {
        case 0:
            return (mem[0][0][1] - mem[0][0][0]) * reciWidth;
        case 1:
            return (mem[0][1][0] - mem[0][0][0]) * reciHeight;
        case 2:
            return (mem[1][0][0] - mem[0][0][0]) * reciDepth;
        }
        return float3_X(NAN, NAN, NAN);
    }
};

template<>
struct DifferenceToUpper<DIM2>
{
    typedef PMacc::math::CT::Int< 1, 1 > OffsetEnd;
    typedef PMacc::math::CT::Int< 0, 0 > OffsetOrigin;

    template<class Memory >
    HDINLINE typename Memory::ValueType operator()(const Memory& mem, const uint32_t direction) const
    {
        const float_X reciWidth = float_X(1.0) / CELL_WIDTH;
        const float_X reciHeight = float_X(1.0) / CELL_HEIGHT;

        switch (direction)
        {
        case 0:
            return (mem[0][1] - mem[0][0]) * reciWidth;
        case 1:
            return (mem[1][0] - mem[0][0]) * reciHeight;

        case 2:
            return float3_X(0., 0., 0.);

        }
        return float3_X(NAN, NAN, NAN);
    }
};

} //namespace picongpu

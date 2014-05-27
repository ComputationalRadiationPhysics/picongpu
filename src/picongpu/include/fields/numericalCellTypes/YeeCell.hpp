/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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
#include "math/vector/Vector.hpp"

namespace picongpu
{
using namespace PMacc;
namespace yeeCell
{

// ___________posE____________
const float_X posE_x_x = 0.5;
const float_X posE_x_y = 0.0;
const float_X posE_x_z = 0.0;

const float_X posE_y_x = 0.0;
const float_X posE_y_y = 0.5;
const float_X posE_y_z = 0.0;

const float_X posE_z_x = 0.0;
const float_X posE_z_y = 0.0;
const float_X posE_z_z = 0.5;

// ___________posB____________
const float_X posB_x_x = 0.0;
const float_X posB_x_y = 0.5;
const float_X posB_x_z = 0.5;

const float_X posB_y_x = 0.5;
const float_X posB_y_y = 0.0;
const float_X posB_y_z = 0.5;

const float_X posB_z_x = 0.5;
const float_X posB_z_y = 0.5;
const float_X posB_z_z = 0.0;

struct YeeCell
{
    /** \tparam floatD_X position of the component in the cell
     *  \tparam DIM3     Fields (E/B) have 3 components, even in 1 or 2D ! */
    typedef ::PMacc::math::Vector<floatD_X,DIM3> VectorVector;

    static HDINLINE VectorVector getEFieldPosition()
    {
#if( SIMDIM == DIM3 )
        const float3_X posE_x(posE_x_x, posE_x_y, posE_x_z);
        const float3_X posE_y(posE_y_x, posE_y_y, posE_y_z);
        const float3_X posE_z(posE_z_x, posE_z_y, posE_z_z);
#elif( SIMDIM == DIM2 )
        const float2_X posE_x(posE_x_x, posE_x_y);
        const float2_X posE_y(posE_y_x, posE_y_y);
        const float2_X posE_z(posE_z_x, posE_z_y);
#endif

        /** position (floatD_x) in cell for E_x, E_y, E_z */
        return VectorVector(posE_x, posE_y, posE_z);
    }

    static  HDINLINE VectorVector getBFieldPosition()
    {
#if( SIMDIM == DIM3 )
        const float3_X posB_x(posB_x_x, posB_x_y, posB_x_z);
        const float3_X posB_y(posB_y_x, posB_y_y, posB_y_z);
        const float3_X posB_z(posB_z_x, posB_z_y, posB_z_z);
#elif( SIMDIM == DIM2 )
        const float2_X posB_x(posB_x_x, posB_x_y);
        const float2_X posB_y(posB_y_x, posB_y_y);
        const float2_X posB_z(posB_z_x, posB_z_y);
#endif

        /** position (floatD_x) in cell for B_x, B_y, B_z */
        return VectorVector(posB_x, posB_y, posB_z);
    }

};


} // yeeCell
} // picongpu


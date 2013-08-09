/**
 * Copyright 2013 Axel Huebl, Heiko Burau, René Widera
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
    typedef ::PMacc::math::Vector<float3_X,DIM3> VectorVector;

       static HDINLINE VectorVector getEFieldPosition() 
    {
        const float3_X posE_x(posE_x_x, posE_x_y, posE_x_z);
        const float3_X posE_y(posE_y_x, posE_y_y, posE_y_z);
        const float3_X posE_z(posE_z_x, posE_z_y, posE_z_z);

        return VectorVector(posE_x, posE_y, posE_z);
    }


       static  HDINLINE VectorVector getBFieldPosition() 
    {
        const float3_X posB_x(posB_x_x, posB_x_y, posB_x_z);
        const float3_X posB_y(posB_y_x, posB_y_y, posB_y_z);
        const float3_X posB_z(posB_z_x, posB_z_y, posB_z_z);

        return VectorVector(posB_x, posB_y, posB_z);
    }

};


} // yeeCell
} // picongpu


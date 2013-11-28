/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * libPMacc is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License and the GNU Lesser General Public License 
 * for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * and the GNU Lesser General Public License along with libPMacc. 
 * If not, see <http://www.gnu.org/licenses/>. 
 */ 
 
#ifndef STLPICFLOAT_HPP
#define STLPICFLOAT_HPP

#include "Vector.hpp"

namespace PMacc
{
namespace math
{

template<int dim>
struct Float : public Vector<float, dim>
{
    HDINLINE Float() {}
    HDINLINE Float(float x) : Vector<float, dim>(x) {}
    HDINLINE Float(float x, float y) : Vector<float, dim>(x,y) {}
    HDINLINE Float(float x, float y, float z) : Vector<float, dim>(x,y,z) {}
    HDINLINE Float(const Vector<float, dim>& vec) : Vector<float, dim>(vec) {}
    
    HDINLINE Float(float1 vec) : Vector<float, dim>(vec.x) {}
    HDINLINE Float(float2 vec) : Vector<float, dim>(vec.x, vec.y) {}
    HDINLINE Float(float3 vec) : Vector<float, dim>(vec.x, vec.y, vec.z) {}
    
    HDINLINE operator float3() const
    {
        BOOST_STATIC_ASSERT(dim == 3);
        return make_float3(this->x(), this->y(), this->z());
    }
};
    
} // math
} // PMacc

#endif // STLPICFLOAT_HPP

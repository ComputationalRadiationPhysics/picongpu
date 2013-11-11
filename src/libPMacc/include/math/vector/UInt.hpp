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
 
#ifndef STLPICUINT_HPP
#define STLPICUINT_HPP

#include "Vector.hpp"

namespace PMacc
{
namespace math
{

template<int dim>
struct UInt : public Vector<uint32_t, dim>
{
    HDINLINE UInt() {}
    HDINLINE UInt(uint32_t x) : Vector<uint32_t, dim>(x) {}
    HDINLINE UInt(uint32_t x, uint32_t y) : Vector<uint32_t, dim>(x,y) {}
    HDINLINE UInt(uint32_t x, uint32_t y, uint32_t z) : Vector<uint32_t, dim>(x,y,z) {}
    HDINLINE UInt(const Vector<uint32_t, dim>& vec) : Vector<uint32_t, dim>(vec) {}
};
    
} // math
} // PMacc

#endif // STLPICUINT_HPP

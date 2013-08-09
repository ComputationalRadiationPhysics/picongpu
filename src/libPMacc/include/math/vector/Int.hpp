/**
 * Copyright 2013 Heiko Burau, René Widera
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
 
#ifndef STLPICINT_HPP
#define STLPICINT_HPP

#include "Vector.hpp"

namespace PMacc
{
namespace math
{

template<int dim>
struct Int : public Vector<int, dim>
{
    HDINLINE Int() {}
    HDINLINE Int(int x) : Vector<int, dim>(x) {}
    HDINLINE Int(int x, int y) : Vector<int, dim>(x,y) {}
    HDINLINE Int(int x, int y, int z) : Vector<int, dim>(x,y,z) {}
    template<typename OtherType>
    HDINLINE Int(const Vector<OtherType, dim>& vec) : Vector<int, dim>(vec) {}
};
    
} // math
} // PMacc

#endif // STLPICINT_HPP

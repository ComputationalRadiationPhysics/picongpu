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
 
#ifndef SIZET_HPP
#define SIZET_HPP

#include "Vector.hpp"

namespace PMacc
{
namespace math
{

template<int dim>
struct Size_t : public Vector<size_t, dim>
{
    HDINLINE Size_t() {}
    HDINLINE Size_t(size_t x) : Vector<size_t, dim>(x) {}
    HDINLINE Size_t(size_t x, size_t y) : Vector<size_t, dim>(x,y) {}
    HDINLINE Size_t(size_t x, size_t y, size_t z) : Vector<size_t, dim>(x,y,z) {}
    HDINLINE Size_t(const Vector<size_t, dim>& vec) : Vector<size_t, dim>(vec) {}
};
    
} // math
} // PMacc

#endif // SIZET_HPP

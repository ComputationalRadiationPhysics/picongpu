/**
 * Copyright 2013-2014 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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

#pragma once

namespace PMacc
{
namespace algorithms
{

namespace math
{

template<typename Type>
struct Floor;

template<typename Type>
struct Float2int_rd;


template<typename T1>
HDINLINE typename Floor< T1>::result floor(T1 value)
{
    return Floor< T1 > ()(value);
}

template<typename T1>
HDINLINE typename Float2int_rd< T1>::result float2int_rd(T1 value)
{
    return Float2int_rd< T1 > ()(value);
}

} //namespace math
} //namespace algorithms
}//namespace PMacc

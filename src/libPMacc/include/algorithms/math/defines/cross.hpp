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
 
/* 
 * File:   math.hpp
 * Author: widera
 *
 * Created on 30. Januar 2013, 09:55
 */

#pragma once

namespace PMacc
{
namespace algorithms
{

namespace math
{

namespace detail
{

template<typename Type1, typename Type2>
struct Cross;

} //namespace detail

template<typename T1, typename T2>
HDINLINE static typename detail::Cross< T1, T2 >::result cross(const T1& value, const T2& value2)
{
    return detail::Cross< T1, T2 > ()(value, value2);
}
} //namespace math
} //namespace algorithms
}//namespace PMacc


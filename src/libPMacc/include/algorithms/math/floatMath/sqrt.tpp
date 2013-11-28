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


#pragma once

#include "types.h"


namespace PMacc
{
namespace algorithms
{
namespace math
{
namespace detail
{

template<>
struct Sqrt<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& value )
    {
        return ::sqrtf( value );
    }
};

template<>
struct RSqrt<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& value )
    {
        return ::rsqrtf( value );
    }
};

} //namespace detail
} //namespace math
} //namespace algorithms
} // namespace PMacc



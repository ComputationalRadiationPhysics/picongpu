/**
 * Copyright 2013 René Widera
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
namespace bmpl = boost::mpl;
namespace pmath = PMacc::math;
namespace pmacc = PMacc;

template<typename T, uint32_t size>
class StaticArray
{
private:
    T data[size];
public:

    template<class> struct result;

    template<class F, typename TKey>
    struct result<F(TKey)>
    {
        typedef T& type;
    };
    
    template<class F, typename TKey>
    struct result<const F(TKey)>
    {
        typedef const T& type;
    };

    HDINLINE
    T& operator[](const int idx)
    {
        return data[idx];
    }

    HDINLINE
    const T& operator[](const int idx) const
    {
        return data[idx];
    }
};

} //namespace PMacc

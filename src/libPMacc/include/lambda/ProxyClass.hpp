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
 
#pragma once

#include "types.h"

namespace PMacc
{
namespace lambda
{
namespace CT
{

template<typename Type, int sizeofType = sizeof(Type), int dummy = 0>
class ProxyClass;

template<typename Type, int sizeofType>
class ProxyClass<Type, sizeofType>
{
private:
    char data[sizeofType];
public:
    typedef Type type;
    
    HDINLINE operator Type&()
    {
        return *(reinterpret_cast<Type*>(this->data));
    }
    
    HDINLINE operator const Type&() const
    {
        return *(reinterpret_cast<const Type*>(this->data));
    }
};

template<typename Type>
class ProxyClass<Type, 0>
{
public:
    typedef Type type;

    HDINLINE operator Type() const
    {
        return Type();
    }
};

template<typename Type>
class ProxyClass<Type, 1>
{
public:
    typedef Type type;

    HDINLINE operator Type() const
    {
        return Type();
    }
};

} // CT
} // lambda
} // PMacc

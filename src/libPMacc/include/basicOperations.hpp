/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, René Widera
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


#include <builtin_types.h>
#include "types.h"
#include <string>
#include <ostream>
#include <math_functions.h>

namespace PMacc
{
namespace
{

DINLINE void atomicAddWrapper(float* address, float value)
{
#if __CUDA_ARCH__ >= 200 // for Fermi, atomicAdd supports floats
    atomicAdd(address, value);
#else
    // float-atomic-add from 
    // [url="http://forums.nvidia.com/index.php?showtopic=158039&view=findpost&p=991561"]http://forums.nvidia.com/index.php?showtop...st&p=991561[/url]
    float old = value;
    while ((old = atomicExch(address, atomicExch(address, 0.0f) + old)) != 0.0f);
#endif
}

DINLINE void atomicAddWrapper(double* inAddress, double value)
{
    uint64_cu* address = (uint64_cu*) inAddress;
    double old = value;
    while (
           (old = __longlong_as_double(atomicExch(address,
                                                  (uint64_cu) __double_as_longlong(__longlong_as_double(atomicExch(address, (uint64_cu) 0L)) +
                                                                                   old)))) != 0.0);
}

}

} //namespace PMacc

/* CUDA STD structs and CPP STD ostream */
template <class T>
std::basic_ostream<T, std::char_traits<T> >& operator<<(std::basic_ostream<T, std::char_traits<T> >& out, const double3& v)
{
    out << "{" << v.x << " " << v.y << " " << v.z << "}";
    return out;
}

template <class T>
std::basic_ostream<T, std::char_traits<T> >& operator<<(std::basic_ostream<T, std::char_traits<T> >& out, const float3& v)
{
    out << "{" << v.x << " " << v.y << " " << v.z << "}";
    return out;
}



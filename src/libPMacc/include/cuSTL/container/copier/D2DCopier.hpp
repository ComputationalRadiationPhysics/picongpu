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
 
#ifndef COPIER_D2DCOPIER_HPP
#define COPIER_D2DCOPIER_HPP

#include "Memcopy.hpp"
#include <types.h>

namespace PMacc
{
namespace copier
{

template<int _dim>
struct D2DCopier
{
    static const int dim = _dim;
    template<typename Type>
    HDINLINE static void copy(Type* dest, const math::Size_t<dim-1>& pitchDest,
         Type* source, const math::Size_t<dim-1>& pitchSource,
         const math::Size_t<dim>& size)
    {
        cudaWrapper::Memcopy<dim>()(dest, pitchDest, source, pitchSource,
                                    size, cudaWrapper::flags::Memcopy::deviceToDevice);
    }         
};

} // copier
} // PMacc

#endif // COPIER_D2DCOPIER_HPP

/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
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

#include <stdint.h>

namespace PMacc
{
namespace assigner
{

template<int dim>
struct HostMemAssigner;

template<>
struct HostMemAssigner<1>
{
    static const int dim = 1;
    template<typename Type>
    static void assign(Type* data, const math::Size_t<dim-1>& pitch, const Type& value,
                       const math::Size_t<dim>& size)
    {
        for(size_t i = 0; i < size.x(); i++) data[i] = value;
    }
};

template<>
struct HostMemAssigner<2u>
{
    static const int dim = 2u;
    template<typename Type>
    static void assign(Type* data, const math::Size_t<dim-1>& pitch, const Type& value,
                       const math::Size_t<dim>& size)
    {
        Type* tmpData = data;
        for(size_t y = 0; y < size.y(); y++)
        {
            for(size_t x = 0; x < size.x(); x++) tmpData[x] = value;
            tmpData = (Type*)(((char*)tmpData) + pitch.x());
        }
    }
};

template<>
struct HostMemAssigner<3>
{
    static const int dim = 3;
    template<typename Type>
    static void assign(Type* data, const math::Size_t<dim-1>& pitch, const Type& value,
                       const math::Size_t<dim>& size)
    {
        for(size_t z = 0; z < size.z(); z++)
        {
            Type* dataXY = (Type*)(((char*)data) + z * pitch.y());
            for(size_t y = 0; y < size.y(); y++)
            {
                for(size_t x = 0; x < size.x(); x++) dataXY[x] = value;
                dataXY = (Type*)(((char*)dataXY) + pitch.x());
            }
        }
    }
};

} // assigner
} // PMacc


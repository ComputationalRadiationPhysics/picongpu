/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera
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

#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/int.hpp>
#include <stdint.h>

namespace PMacc
{
namespace assigner
{

namespace bmpl = boost::mpl;

template<typename T_Dim = bmpl::_1, typename T_CartBuffer = bmpl::_2>
struct HostMemAssigner;

template<typename T_CartBuffer>
struct HostMemAssigner<bmpl::int_<1>, T_CartBuffer>
{
    BOOST_STATIC_CONSTEXPR int dim = 1;
    typedef T_CartBuffer CartBuffer;

    template<typename Type>
    HINLINE void assign(const Type& value)
    {
        // "Curiously recurring template pattern"
        CartBuffer* buffer = static_cast<CartBuffer*>(this);

        for(size_t i = 0; i < buffer->size().x(); i++)
            buffer->dataPointer[i] = value;
    }
};

template<typename T_CartBuffer>
struct HostMemAssigner<bmpl::int_<2>, T_CartBuffer>
{
    BOOST_STATIC_CONSTEXPR int dim = 2;
    typedef T_CartBuffer CartBuffer;

    template<typename Type>
    HINLINE void assign(const Type& value)
    {
        // "Curiously recurring template pattern"
        CartBuffer* buffer = static_cast<CartBuffer*>(this);

        Type* tmpData = buffer->dataPointer;
        for(size_t y = 0; y < buffer->size().y(); y++)
        {
            for(size_t x = 0; x < buffer->size().x(); x++)
                tmpData[x] = value;
            tmpData = reinterpret_cast<Type*>(reinterpret_cast<char*>(tmpData) + buffer->pitch.x());
        }
    }
};

template<typename T_CartBuffer>
struct HostMemAssigner<bmpl::int_<3>, T_CartBuffer>
{
    BOOST_STATIC_CONSTEXPR int dim = 3;
    typedef T_CartBuffer CartBuffer;

    template<typename Type>
    HINLINE void assign(const Type& value)
    {
        // "Curiously recurring template pattern"
        CartBuffer* buffer = static_cast<CartBuffer*>(this);

        for(size_t z = 0; z < buffer->size().z(); z++)
        {
            Type* dataXY = reinterpret_cast<Type*>(reinterpret_cast<char*>(buffer->dataPointer) + z * buffer->pitch.y());
            for(size_t y = 0; y < buffer->size().y(); y++)
            {
                for(size_t x = 0; x < buffer->size().x(); x++)
                    dataXY[x] = value;
                dataXY = reinterpret_cast<Type*>(reinterpret_cast<char*>(dataXY) + buffer->pitch.x());
            }
        }
    }
};

} // assigner
} // PMacc


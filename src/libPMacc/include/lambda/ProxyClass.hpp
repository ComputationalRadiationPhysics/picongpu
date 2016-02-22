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

#include "pmacc_types.hpp"

namespace PMacc
{
namespace lambda
{
namespace CT
{
namespace detail
{
    /** Create a same-sized type we can savely cast to
     *  Raw data represents a contructor-free data block
     */
    template<uint32_t x>
    struct raw_data
    {
        uint8_t data[x];
    };
} // namespace detail

template<typename Type, int sizeofType = sizeof(Type), int dummy = 0>
class ProxyClass;

template<typename Type, int sizeofType>
class ProxyClass<Type, sizeofType>
{
private:
    detail::raw_data<sizeofType> data;
public:
    typedef Type type;

    HDINLINE operator Type&()
    {
        return *(reinterpret_cast<Type*>(&data));
    }

    HDINLINE operator const Type&() const
    {
        return *(reinterpret_cast<const Type*>(&data));
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

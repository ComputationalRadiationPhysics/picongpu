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
#include "RefWrapper.hpp"

namespace PMacc
{

/** functor to get forwarded value
 *
 * @tparam type of forwarded value
 * @tparam[out] ::type result of functor
 */
template<typename T_Type>
struct GetForwardedValue
{
    typedef const T_Type& type;

    HDINLINE type operator()(type value)
    {
        return value;
    }
};

/** specialization for forwarded RefWrapper */
template<typename T_Type>
struct GetForwardedValue<RefWrapper<T_Type> >
{
    typedef T_Type& type;

    HDINLINE type operator()(type value)
    {
        return value;
    }
};

/** get forwarded value
 *
 * @tparam arbitrary type of parameter
 * 
 * @param arg forwarded value
 * @return reference to original value
 */
template<typename T_Type>
HDINLINE static typename GetForwardedValue<T_Type>::type
getForwardedValue(const T_Type& arg)
{
    return GetForwardedValue<T_Type>()(arg);
}


/** function to forward changeable value as const variable
 *
 * any variable which should changeable in functor but must passed over
 * constant helper functor classes are packed in a constant
 * type and can passed (forwarded) to functor
 *
 * To get original forwarded type back @see getForwardedValue
 */
template<typename T_Type>
HDINLINE static const T_Type&
forward(const T_Type& arg)
{
    return arg;
}

template<typename T_Type>
HDINLINE static RefWrapper<T_Type>
forward(T_Type& arg)
{
    return arg;
}

} // namespace PMacc
